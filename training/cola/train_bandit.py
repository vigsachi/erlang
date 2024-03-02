import os
import sys
sys.path.insert(1, os.getcwd())

import time
import copy
import pickle
import random
import logging
import datetime
import operator
import numpy as np
from tqdm import tqdm

from google.cloud import trace_v1
from google.protobuf.timestamp_pb2 import Timestamp 

from utils import kube as kube_utils
from utils import cluster as cluster_utils
from utils import hpa as hpa_utils
from training.cola.bandit_algo import UCB_Bandit
from utils.locust_loadgen import LoadGenerator
from utils.launch_apps import launch_application
from inference.utils.cloud_metrics import CloudMetrics

logging.basicConfig(filename='logs/training.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('training')

# Class to Train Bandit Autoscaling algorithm.
class BanditTrainer(object):
    def __init__(self, name, config, train_config, search_history_len=3, hatch_rate=100, restart_policy=False):

        # Initialize the configs for the deployment and training procedure.
        self.config = config
        self.train_config = train_config
        self.train_config.app_config = config
        self.all_res = []
        
        # Set up action space.
        self.num_actions = {}
        for service, max_replicas in config.deployments.items():
            self.num_actions[service] = max_replicas        
                
        # Create a place to store configs.
        self.restart_policy = restart_policy
        self.policy = {}
        for service in config.services:
            self.policy[service] = 1    
        kube_utils.apply_policy(self.policy)
        logger.info('Applied policy {}'.format(self.policy))

        # Services we searched recently.
        self.search_history = []
        self.search_history_len = 2 # Number of recently searched services to exclude for next search.
        
        # Model name and persistance details.
        self.name = name
        self.save_path = os.path.join('models', self.train_config.app_config.name, self.name)
        self.create_save_dirs()

        # Setup load generator for sampling qoe.
        self.hatch_rate = hatch_rate
        #embed()

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
        print('weights: {}, {}'.format(self.train_config.w_l, self.train_config.w_i))

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


        # Disable cluster autoscaling.
        cluster_utils.disable_node_pool_autoscaling(cluster=self.train_config.cluster_name,
                                                    project=self.train_config.project_name,
                                                    zone=self.train_config.zone,
                                                    node_pool=self.train_config.node_pool)
        logger.info('Disabled node pool autoscaling for {}'.format(self.train_config.node_pool))

        # Launch the application.
        launch_application(config=self.config)
        time.sleep(90)

    def run(self):

        # Train each context requested for the application.
        for context in self.train_config.train_rps:
            logger.info("Training context: {}".format(context))
            self.train_context(context)

        # Clean up node pool after completion.
        self.cleanup()
        return

    def create_context_map(self):

        cluster_utils.scale_node_pool(
                                       cluster=self.train_config.cluster_name,
                                       project=self.train_config.project_name,
                                       zone=self.train_config.zone,
                                       node_pool=self.train_config.node_pool,
                                       num_nodes=sum(self.config.deployments.values())
                                   )
        print("Scaled to {} instances".format(sum(self.config.deployments.values())))
        kube_utils.apply_policy(self.config.deployments)
        time.sleep(30)

        # Construct object to query cloud metrics.
        self.cm = CloudMetrics(self.train_config.project_name, self.train_config.req_names)

        # Train each context requested for the application.
        ctx_map = {}
        for context in self.train_config.train_rps:
            logger.info("Training context: {}".format(context))
            ctx_map[context] = self.record_context(context)

        # Save the context map.
        self.save_context_map(ctx_map)

        # Clean up node pool after completion.
        self.cleanup()
        return

    def train_context(self, context):

        # Apply our current policy.
        self.all_res = []

        # Restart learning process for each policy.
        if self.restart_policy is True:
            logger.info('Restarting policy for each context')
            self.policy = {}
            for service in self.config.services:
                self.policy[service] = 1    
            kube_utils.apply_policy(self.policy)
            logger.info('Applied policy {}'.format(self.policy))

        # Keep policy from previous learned context.
        else:
            kube_utils.apply_policy(self.policy)
            logger.info('Applied policy {}'.format(self.policy))

        # Record start time for training.
        start = time.time()
        
        # Warm start the training if requested.
        if self.warm_start is True:
            self.policy = self.warm_start(self.train_config.cpu_t, context, self.train_config.cpu_duration)

        # Run training iterations.
        for _iter in tqdm(range(self.train_config.train_iters)):
            
            # Get service to scale for this iteration based on our search strategy.
            service = self.select_service_to_scale(context, self.train_config.search_strategy)
            logger.info("Selected {service} to scale".format(service=service))

            # Run the bandit to evaluate best replicas for the given service.
            ucb = self.run_bandit(service, context)
            self.all_res += ucb.all_res

            # Get opt action for service.
            opt_action = np.argmax(ucb.k_reward)
            opt_replicas = ucb.action_to_rep_map[opt_action]
            self.policy[service] = opt_replicas

            # Scale service to opt_replicas.
            kube_utils.apply_policy(self.policy)
            logger.info("Set {service} to {opt_replicas} replicas".format(service=service, opt_replicas=opt_replicas))
            logger.info("Current policy {}".format(self.policy))
            
            # Save configs and models for this search.
            self.save_iter(ucb, self.policy, context, _iter)
            self.save_configs()
            
            # Check for early stop.
            #embed()
            if ucb.k_latency[np.argmax(ucb.k_reward)] <= self.train_config.latency_threshold:
                print("Stopping early. Policy {}".format(self.policy))
                break



        # Reduce weight parameter and retry context.
        #if ucb.k_latency[np.argmax(ucb.k_reward)] > self.train_config.latency_threshold:
        #    if self.train_config.w_i > 1:
        #        self.train_config.w_i -= 1
        #        self.lg = LoadGenerator(host=self.config.host,
        #                        locustfile=self.train_config.locustfile,
        #                        duration=self.train_config.sample_duration,
        #                        csv_path=self.train_config.csv_path,
        #                        latency_threshold=self.train_config.latency_threshold,
        #                        w_i=self.train_config.w_i,
        #                        w_l=self.train_config.w_l,
        #                        hatch_rate=self.hatch_rate,
        #                        lat_opt=self.train_config.lat_opt,
        #                        )
        #        self.train_context(context)


        # Record end time for training.
        end = time.time()
        self.policy['time'] = end - start
        self.policy['all_res'] = copy.deepcopy(self.all_res)

        # Save final config for context and the training/app configs that we used.
        self.save_configs()
        self.save_iter(None, self.policy, context, 'final') # Save final policy
        logger.info("Final Config for context {}: {}".format(context, self.policy))

        # Reset search history
        self.search_history = []

        return

    def record_context(self, context, duration=240):

        # Generate load.
        self.lg.generate_load(rps=context, duration=duration)

        # Get most recent rps and rps per operation (our context).
        #try:
        rps = self.cm.get_request_count(minutes=int(duration/60), pod_filter=self.train_config.pod_filter)
        #except:
        #    embed()
        req_vector, req_index = self.cm.get_request_dist(minutes=int(duration/60))
        logger.info('Config for context {}: {}'.format(context, {'rps': rps, 'req_vector': req_vector, 'req_index': req_index}))
        return {'rps': rps, 'req_vector': req_vector, 'req_index': req_index}


    def warm_start(self, cpu_t, context, duration):
        ''' Warm start the training config with a CPU based autoscaler. '''

        # Update cpu policy, apply load and delete the cpu policy.
        hpa_utils.update_autoscaling_policy(config=self.config, cpu_t=cpu_t) # Add autoscaler.
        self.lg.generate_load(rps=context) # Generate load.
        hpa_utils.delete_autoscaling_policy(config=self.config) # Delete autoscaling policy.


        # Fetch the current config and return it.
        policy = kube_utils.get_current_deployments()
        return policy


    def select_service_to_scale(self, context, strategy='cpu'):
        ''' Select service to scale. '''

        if strategy == 'cpu':
            # Choose next service by highest cpu utilization for the current policy na dload.

            # Baseline CPU usage.
            util_df_pre = kube_utils.get_pod_statistics(self.config.cpu_requests, self.config.mem_requests)
            util_df_pre = util_df_pre[util_df_pre[3].isin(self.config.deployments.keys())]
            util_df_pre = util_df_pre[~util_df_pre[3].isin(self.search_history)]

            # Generate the load.
            self.lg.generate_load(rps=context, duration=90)
            time.sleep(10)

            # Get utilization metrics averaged across pods.
            util_df = kube_utils.get_pod_statistics(self.config.cpu_requests, self.config.mem_requests)

            # Exclude recently searched services
            util_df = util_df[util_df[3].isin(self.config.deployments.keys())]
            util_df = util_df[~util_df[3].isin(self.search_history)]

            # Select service based on highest cpu usage increase.
            max_serv_idx = np.argmax(util_df.sort_values([3])[1] - util_df_pre.sort_values([3])[1])
            service =  util_df.sort_values([3])[3].iloc[max_serv_idx]

        if strategy == 'mem':
            # Choose next service by highest cpu utilization for the current policy na dload.
            logger.info('Selected service to scale by mem')

            # Baseline CPU usage.
            util_df_pre = kube_utils.get_pod_statistics(self.config.cpu_requests, self.config.mem_requests)
            util_df_pre = util_df_pre[util_df_pre[3].isin(self.config.deployments.keys())]
            util_df_pre = util_df_pre[~util_df_pre[3].isin(self.search_history)]

            # Generate the load.
            self.lg.generate_load(rps=context, duration=90)
            time.sleep(10)

            # Get utilization metrics averaged across pods.
            util_df = kube_utils.get_pod_statistics(self.config.cpu_requests, self.config.mem_requests)

            # Exclude recently searched services
            util_df = util_df[util_df[3].isin(self.config.deployments.keys())]
            util_df = util_df[~util_df[3].isin(self.search_history)]

            # Select service based on highest cpu usage increase.
            max_serv_idx = np.argmax(util_df.sort_values([3])[2] - util_df_pre.sort_values([3])[2])
            service =  util_df.sort_values([3])[3][max_serv_idx]


        if strategy == 'random':
            logger.info('Selected service to scale randomly')

            # Choose next service randomly.
            service = random.choice(self.config.services)

        if strategy == 'ordered':
            # Choose next service in the order of the service list provided.

            if len(self.search_history) == 0:
                service = self.config.services[0]
            else:
                # Get index of last service we searched.
                last_idx = self.config.services.index(self.search_history[-1])
                service = self.config.services[0] if last_idx == len(self.config.services) else self.config.services[last_idx + 1]

        if strategy == 'longest_span':
            logger.info('Selected service to scale by longest average span')

            # Record start time.
            start = datetime.datetime.now()

            # Generate load.
            self.lg.generate_load(rps=context, duration=90)
            time.sleep(10)

            # Record start time.
            end = datetime.datetime.now()

            # Get traces.
            traces = get_traces(start, end)
            sorted_services = sort_traces_by_service_length(traces, sort_by='length')

            # Get service to scale.
            sorted_services = [x for x in sorted_services if x in self.config.deployments.keys()]
            service = sorted_services[0]


        if strategy == 'most_frequent_span':
            logger.info('Selected service to scale by most frequent span')

            # Record start time.
            start = datetime.datetime.now()

            # Generate load.
            self.lg.generate_load(rps=context, duration=90)
            time.sleep(10)

            # Record start time.
            end = datetime.datetime.now()

            # Get traces.
            traces = get_traces(start, end)
            sorted_services = sort_traces_by_service_length(traces, sort_by='count')

            # Get service to scale.
            sorted_services = [x for x in sorted_services if x in self.config.deployments.keys()]
            service = sorted_services[0]



        # Add service to search history.
        self.search_history.append(service)
        self.search_history = self.search_history[-self.search_history_len:]

        return service


    def run_bandit(self, service, context, action_multiple = 2, max_iters = 50, k=5, strategy='ucb1'):
        ''' Choose best arm (num replicas) for the given service '''

        # Identify number of arms and iterations for the bandit.
        current_action = self.policy[service]
        ucb_iters = min(max_iters,int(k*action_multiple)) # Heuristic for search iterations we need (roughly 4x the replica range)

        # Create UCB Bandit for current service..
        ucb = UCB_Bandit(k=k,
                        c=self.train_config.c, 
                        iters=ucb_iters, 
                        service=service,
                        context=context,
                        lg=self.lg,
                        policy=self.policy, 
                        current_action=current_action,
                        num_actions=self.num_actions[service],
                        lat_opt=self.train_config.lat_opt,
                        strategy=strategy)

        # Run experiments
        ucb.run()

        return ucb


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

    def save_iter(self, ucb, policy, context, _iter):

        # Save bandit model.
        if ucb:
            bandit_path = os.path.join(self.bandit_path, str(context), 'bandit_{}.pk'.format(_iter))
            pickle.dump(copy.deepcopy(ucb), open(bandit_path, "wb"))


        # Save policy.
        if policy:
            policy_path = os.path.join(self.policy_path, str(context), 'policy_{}.pk'.format(_iter))
            pickle.dump(copy.deepcopy(policy), open(policy_path, "wb"))

        # Save model and config to the train config.
        self.train_config.add_context(context, policy, ucb)

        return


    def save_configs(self):

        # Save application config.
        app_config_path = os.path.join(self.save_path, 'app_config.pk')
        pickle.dump(self.config, open(app_config_path, "wb"))

        # Save training config.
        train_config_path = os.path.join(self.save_path, 'train_config.pk')
        pickle.dump(self.train_config, open(train_config_path, "wb"))

        return

    def save_context_map(self, context_map):

        # Save context map.
        context_map_path = os.path.join(self.save_path, 'context_map.pk')
        pickle.dump(context_map, open(context_map_path, "wb"))

        return


    def check_early_stop(self, context, duration):

        # Generate load and get the end to end statistics.
        time.sleep(15) # Wait for the policy applied before early stop was called to get enacted.
        self.lg.generate_load(rps=context, duration=duration)
        stats = self.lg.read_load_statistics(context)

        # Check if we should stop early.
        early_stop = False
        if stats[self.train_config.lat_op] <= self.train_config.latency_threshold:
            early_stop = True

        logger.info("Early stop was {}, stats {}".format(early_stop, stats))
        return early_stop

    def cleanup(self):

        # Scale down training node pool.
        cluster_utils.scale_node_pool(
                                        cluster=self.train_config.cluster_name,
                                        project=self.train_config.project_name,
                                        zone=self.train_config.zone,
                                        node_pool=self.train_config.node_pool,
                                        num_nodes=0
                                    )
        return





def get_traces(start, end, limit=1000):
    # Create Protobufs for start and end times.
    start_proto = Timestamp() 
    start_proto.FromDatetime(start) 
    end_proto = Timestamp() 
    end_proto.FromDatetime(end) 
    print(start_proto)
    print(end_proto)

    # Create Client.
    client = trace_v1.TraceServiceClient() 

    # Iterate over all traces.
    all_traces = {}
    for i, trace in enumerate(tqdm(client.list_traces(project_id='vig-cloud', start_time=start_proto, end_time=end_proto))): 
        all_traces[i] = client.get_trace(project_id='vig-cloud', trace_id=trace.trace_id)
        if len(all_traces) > limit:
            break

    return all_traces




def sort_traces_by_service_length(trace_list, sort_by='duration'):

    all_traces = list([x for x in trace_list.values()])
    durations = {}

    for trace in all_traces:
        for span in trace.spans:
            svc_type = span.name.split('.')[-2].lower()
            if svc_type not in durations:
                durations[svc_type] = []

            durations[svc_type].append((span.end_time.ToDatetime() - span.start_time.ToDatetime()).total_seconds()*1000)

    all_durations = {}
    if sort_by == 'duration':
        all_durations =  {k:np.mean(v) for k,v in durations.items()}
    if sort_by == 'count':
        all_durations =  {k:len(v) for k,v in durations.items()}

    sorted_durations = [x[0] for x in sorted(durations.items(), key=operator.itemgetter(1), reverse=True)]

    return sorted_durations


