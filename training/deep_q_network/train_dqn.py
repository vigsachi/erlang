import os
import sys
sys.path.insert(1, os.getcwd())

import time
import pickle
import logging
import numpy as np
from copy import deepcopy

from ddpg.ddpg import DDPG
from ddpg.evaluator import Evaluator

import utils.kube as kube_utils
import utils.cluster as cluster_utils
from utils.kube import get_current_deployments
from utils.locust_loadgen import LoadGenerator
from utils.launch_apps import launch_application


logging.basicConfig(filename='logs/training.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('training')

# Class to Train Bandit Autoscaling algorithm.
class DQNTrainer(object):
    def __init__(self, name, config, train_config, search_history_len=3, hatch_rate=100, num_samples=100):

        # Initialize the configs for the deployment and training procedure.
        self.config = config
        self.train_config = train_config
        self.train_config.app_config = config
        self.num_samples = num_samples
        
        # Set up action space.
        self.num_actions = {}
        for service, max_replicas in config.deployments.items():
            self.num_actions[service] = max_replicas        
                
        # Create a place to store configs.
        self.policy = {}
        for service in config.services:
            self.policy[service] = 1    
        kube_utils.apply_policy(self.policy)
        logger.info('Applied policy {}'.format(self.policy))

        # Services we searched recently.
        self.search_history = []
        self.search_history_len = search_history_len # Number of recently searched services to exclude for next search.
        
        # Model name and persistance details.
        self.name = name
        self.save_path = os.path.join('models', self.train_config.app_config.name, self.name)
        self.create_save_dirs()

        # Setup load generator for sampling qoe.
        self.hatch_rate = hatch_rate

        self.lg = LoadGenerator(host=self.config.host, 
                                locustfile=self.train_config.locustfile, 
                                duration=self.train_config.sample_duration,
                                csv_path=self.train_config.csv_path,
                                latency_threshold=self.train_config.latency_threshold,
                                w_i=self.train_config.w_i,
                                w_l=self.train_config.w_l,
                                hatch_rate=self.hatch_rate,
                                lat_opt=self.train_config.lat_opt,
                                )

        # Setup the node pool for training
        cluster_utils.scale_node_pool(
                                        cluster=self.train_config.cluster_name,
                                        project=self.train_config.project_name,
                                        zone=self.train_config.zone,
                                        node_pool=self.train_config.node_pool,
                                        num_nodes=self.train_config.max_nodes
                                    )
        logger.info('Scaled node pool {} to {} nodes'.format(self.train_config.node_pool, 
                                                              self.train_config.max_nodes))

        # Launch the application, wait for pods to come online.
        launch_application(config=self.config)
        time.sleep(90)

        # Get model training args.
        self.args = construct_training_args()

    def run(self):

        # Record start time.
        self.start_train_time = time.time()

        # Initialize Model.
        self.init_model()

        # Run training.
        for context in self.train_config.train_rps:
            self.train(context, tot_steps = int(self.num_samples / len(self.train_config.train_rps)))

        # Record end time.
        self.end_train_time = time.time()

        # Tear down cluster.
        self.cleanup()


    def init_model(self):
        """ Create ddpg model. """

        # Construct the states and actions for our environment.
        mcs_ranges = {}
        for deployment in self.config.deployments:
            mcs_ranges[deployment] = np.arange(1, self.config.deployments[deployment])

        # Compose entire search space.
        self.search_coords = list(mcs_ranges.keys())
        search_space = self.train_config.train_rps + list(mcs_ranges.values())
        #self.training_points = np.array(np.meshgrid(search_space).T.reshape(-1, len(search_space)))

        # Create the state space.
        self.nb_states = len(self.search_coords)*3 + 1 # Replicas, CPU, MEM for each service + context.
        self.nb_actions = len(self.search_coords) # Number of replicas for each service.

        # Create Agent.
        self.agent = DDPG(self.nb_states, self.nb_actions, self.args)
        self.evaluator = Evaluator(self.args['validate_episodes'], 
            self.args['validate_steps'], self.save_path, 
            max_episode_length=self.args['max_episode_length'])

        self.res = []

        return
    
    def train(self, context, tot_steps=100):

        # Initialization
        self.agent.is_training = True
        step = episode = episode_steps = 0
        episode_reward = 0.
        warmup_steps = 1
        observation = None
        print("Training context {} for {} steps".format(context, tot_steps))

        while step < tot_steps:

            # Get observation.
            observation = self.get_state(context)
            observation = deepcopy(observation)

            # Select action.
            if step <= warmup_steps:
                action = self.agent.random_action()
            else:
                action = self.agent.select_action(observation)

            # Take step and receive next observation, reward.
            reward, observation2, done = self.step(action, context)
            observation2 = deepcopy(observation2)

            if self.args['max_episode_length'] and episode_steps >= self.args['max_episode_length'] -1:
                done = True

            # Give agent observation, reward and update the policy.
            self.agent.observe(reward, np.array(observation2), done)
            if step > warmup_steps:
                try:
                    self.agent.update_policy()
                except:
                    pass

            # Save intermediate model.
            self.agent.save_model(self.save_path)

            # Update counters.
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            # Print debug message with reward.
            if self.args['debug']: 
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(episode_steps, 
                episode_reward / float(episode_steps)))

            if done: # end of episode
                if self.args['debug']: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

                self.agent.memory.append(
                    observation,
                    self.agent.select_action(observation),
                    0., False
                )

                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
            

        return

    def get_state(self, context):

        # Get information on number of deployments, CPU and Memory utilization.
        deployments = get_current_deployments()
        util_df = kube_utils.get_pod_statistics(self.config.cpu_requests, self.config.mem_requests)
        print(util_df)
        util_df = util_df[util_df[3].isin(self.config.deployments.keys())]

        # Create current state with information on number of replicas, CPU % and MEM %.
        state = []
        for deployment in self.search_coords:
            state.append(deployments[deployment]) # Number of replicas

            util_row = util_df[util_df[3].isin([deployment])].iloc[0]
            state.append(util_row[1]) # CPU utilization (%)
            state.append(util_row[2]) # MEM utilization (%)
        state.append(context)

        return state
        
    def step(self, action, context):

        # Convert action to deployment config.
        prYellow(action)
        deployment_config = {}
        for i, num_replicas in enumerate(action):
            max_replicas = self.config.deployments[self.search_coords[i]]
            deployment_replicas = (num_replicas + 1.0) * max_replicas / 2 # Convert from [-1,1] to [0, max_rep]
            deployment_replicas = np.round(deployment_replicas)
            deployment_replicas = min(deployment_replicas, self.config.deployments[self.search_coords[i]])
            deployment_replicas = max(deployment_replicas, 1)

            deployment_config[self.search_coords[i]] = int(deployment_replicas)
        prYellow(deployment_config)

        # Scale deployment config.
        kube_utils.apply_policy(deployment_config)
        time.sleep(45)

        # Evaluate action.
        self.lg.generate_load(rps=context)
        reward, stats = self.lg.eval_qoe(rps=context, action=sum(deployment_config.values()))

        # Record results.
        self.res.append({'action': action, 'reward': reward, 'stats': stats})

        # Get observation currently.
        observation = self.get_state(context)

        return reward, observation, False

    def cleanup(self):
        # Scale down training node pool.
        cluster_utils.scale_node_pool(
                                        cluster=self.train_config.cluster_name,
                                        project=self.train_config.project_name,
                                        zone=self.train_config.zone,
                                        node_pool=self.train_config.node_pool,
                                        num_nodes=0
                                    )

        # Save start and end times.
        train_time_path = os.path.join(self.save_path, 'training_time.pk')
        pickle.dump({'start': self.start_train_time, 'end': self.end_train_time}, open(train_time_path, "wb"))

        # Save model.
        model_path = os.path.join(self.policy_path, 'policy.pk')
        pickle.dump(self, open(model_path, "wb"))

        return


    def create_save_dirs(self):

        # Make directory for the top level training routine.
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)

        # Make directory to save policy updates.
        self.policy_path = os.path.join(self.save_path, 'policy')
        if not os.path.exists(self.policy_path): 
            os.mkdir(self.policy_path)

        # Make directory to save bandit updates
        self.bandit_path = os.path.join(self.save_path, 'bandit')
        if not os.path.exists(self.bandit_path): 
            os.mkdir(self.bandit_path)

        # Create subfolder for each rps value we train on.
        for rps in self.train_config.train_rps:
            if not os.path.exists(os.path.join(self.policy_path, str(rps))):
                os.mkdir(os.path.join(self.policy_path, str(rps)))
            if not os.path.exists(os.path.join(self.bandit_path, str(rps))):
                os.mkdir(os.path.join(self.bandit_path, str(rps)))
        return



def construct_training_args():

    return {
            'mode': 'train',
            'hidden1': 200,
            'hidden2': 80,
            'rate': .001,
            'prate': 0.0001,
            'warmup': 5,
            'discount': 0.99,
            'bsize': 64,
            'rmsize': 6000000,
            'window_length': 1,
            'tau': .001,
            'ou_theta': 0.15,
            'ou_sigma': 0.2,
            'ou_mu': 0.0,
            'validate_episodes': 20,
            'max_episode_length': 400,
            'validate_steps': 5,
            'output': 'output',
            'debug': 'debug',
            'init_w': .003,
            'train_iter': 100,
            'epsilon': 50000,
            'seed': 1,
            'resume': 'default',
            }

def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
