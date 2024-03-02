import time
import math
import logging
import numpy as np 
from tqdm import tqdm

import utils.kube as kube_utils

logging.basicConfig(filename='logs/training.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('training')


class UCB_Bandit:
    '''
    Upper Confidence Bound Bandit
    '''
    def __init__(self, k, c, iters, service, context, lg, policy, current_action, num_actions, lat_opt, strategy='ucb1'):
        
        # Bandit settings and statistics.
        self.k = k # Number of arms
        self.c = c # Exploration parameter
        self.iters = iters # Number of iterations
        self.n = 1 # Step count
        self.k_n = np.ones(k) # Step count for each arm
        self.mean_reward = 0 # Total mean reward
        self.reward = np.zeros(iters) # Reward for each iter
        self.k_reward = np.zeros(k) # Mean reward for each arm
        self.strategy = strategy # Strategy to use to select optimal action
        self.all_res = []

        
        # Mean Latency, Failures for each arm.
        self.k_latency = np.zeros(k)
        self.k_failures = np.zeros(k)
        self.policy = policy # Store policy for configuration
        self.current_action = current_action
        self.num_actions = num_actions
        self.k_latency_dict = {i:[] for i in range(k)}
        self.lat_opt = lat_opt

        # Service we are scaling and associated parameters.
        self.service = service
        self.context = context
        self.lg = lg

        self.action_to_rep_map = self.create_action_to_rep_map(current_action, num_actions, k)
        
    def create_action_to_rep_map(self, current_action, num_actions, k):

        action_to_rep_map = {}

        actions = [current_action+x for x in list(range(k))]
        actions = [action - math.floor(k/2) for action in actions]

        # Check if underflow.
        if min(actions) < 1:
            actions = [x+abs(1-min(actions)) for x in actions]

        # Check for overflow.
        elif max(actions) > num_actions:
            actions = [x-(max(actions)-num_actions) for x in actions]
            
        for i, action in enumerate(actions):
            action_to_rep_map[i] = action

        return action_to_rep_map
                         

    def pull(self, iter_, strategy):
        
        # Select action according to UCB Criteria.
        if strategy == 'ucb1':
            # Select action according to UCB Criteria.
            a = np.argmax(self.k_reward + self.c * np.sqrt(
                    (np.log(self.n)) / self.k_n))
        else:
            a = iter_ % self.k       
            
        # Scale deployment based on action.
        kube_utils.scale_deployment(self.service, self.action_to_rep_map[a])
        time.sleep(30) # Allow the deployment adjust to the new number of replicas.

        # Get reward based on action taken.
        self.lg.generate_load(rps=self.context)
        reward, stats = self.lg.eval_qoe(rps=self.context, action=a+1)

        logger.info("For context {} and action {} got reward {}, res {}".format(
            self.context, self.action_to_rep_map[a], reward, stats))

        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        self.k_latency[a] = self.k_latency[a] + (
            stats[self.lat_opt] - self.k_latency[a]) / self.k_n[a]
        self.k_failures[a] = self.k_failures[a] + (
            stats['Failures RPS'] - self.k_failures[a]) / self.k_n[a]
        self.k_latency_dict[a].append(stats[self.lat_opt])        

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update results storage.
        total_replicas = retry(kube_utils.get_current_deployments)
        reward_tot_replicas = self.lg.compute_qoe(stats, sum(total_replicas.values()))
        self.all_res.append({'stats':stats, 'total_replicas':total_replicas, 'reward_total_replicas':reward_tot_replicas})

        return
        
        
    def run(self):
        for i in tqdm(range(self.iters)):
            retry(self.pull(i, self.strategy))
            self.reward[i] = self.mean_reward
        return
    
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        
        # Reset reward stats.
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        
        # Reset arm stats.
        self.k_reward = np.zeros(self.k)
        self.k_latency = np.zeros(self.k)
        self.k_failures = np.zeros(self.k)
        
        return



import time
def retry(fun, max_tries=10):
    for i in range(max_tries):
        try:
           time.sleep(0.3) 
           return fun()
        except Exception:
            continue

    return -1
