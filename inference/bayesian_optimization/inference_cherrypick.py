import os
import sys
sys.path.insert(1, '..')

import time
import copy
import math
import pickle
import logging
import matplotlib.pyplot as plt

import utils.hpa as hpa_utils
import utils.kube as kube_utils
import utils.cluster as cluster_utils
from inference.utils.cloud_metrics import CloudMetrics
from training.bayesian_optimization.train_cherrypick import sample_round_multivariate
import argparse


logging.basicConfig(filename='logs/scaling.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('scaling')

class CherryPickAutoscaler(object):
    ''' Single model based inference on autoscaling policy'''

    def __init__(self, train_config_path, pod_filter, init_nodes, add_nodes = 0, second_context=None):

        # Save the train config path.
        self.train_config_path = train_config_path

        # Load the model from the given path.
        self.load_model()

        # Construct object to query cloud metrics.
        self.cm = CloudMetrics(self.train_config.project_name, self.train_config.req_names)

        # Store current context.
        self.current_rps = 0
        self.second_context = second_context
        self.rps_history = []
        self.req_vector = None
        self.req_index = None
        self.current_nodes = init_nodes
        self.current_policy = None
        self.pod_filter = pod_filter
        self.add_nodes = add_nodes
        logger.info('Started autoscaler.')
        
        # Start off autoscaling and fall back to cpu if beyond training range.
        self.scaling = True
        
    def load_model(self):

        # Load context map.
        self.context_map = pickle.load(open(os.path.join(self.train_config_path, 'context_map.pk'), "rb"))
        self.train_config = pickle.load(open(os.path.join(self.train_config_path, 'train_config.pk'), "rb"))
        self.train_config.config_map[0] = {service:1 for service in self.train_config.app_config.services} # Update train_config map with 0 rps value (set to min replica for all services)

        
        # Create rps map.
        self.rps_map = {0:0}
        self.rps_rev_map = {0:0}
        for k,v in self.context_map.items():
            self.rps_map[k] = self.context_map[k]['rps']
            self.rps_rev_map[self.context_map[k]['rps']] = k

        # Load training config.
        self.model = copy.deepcopy(pickle.load(open(os.path.join(self.train_config_path, 'policy/policy.pk'), "rb")))

        # Create list of train config.
        self.train_context = sorted(list(self.rps_map.keys()))

        logger.info("Have models for following context: {}".format(self.train_context))
        logger.info("Using path: {}".format(self.train_config_path))

        # Load app config.
        self.app_config = self.train_config.app_config


    def run(self, num_iters=1e20, wait_time=5):

        for _iter in range(num_iters):

            # Get new metrics.
            scale = self.update_metrics()
            if scale is False:
                continue

            # Get pod policy.
            scale_config = self.get_pod_config()

            # Scale pods and nodes.
            self.scale_cluster(scale_config)

            # Wait before repeating process.
            time.sleep(wait_time)

    def update_metrics(self, minutes=5):

        # Get most recent rps and rps per operation (our context).
        try:
            rps = self.cm.get_request_count(minutes=minutes, pod_filter=self.pod_filter)
        except:
            return False

        req_vector, req_index = self.cm.get_request_dist(minutes=minutes)
        logger.info("Current Workload RPS: {}".format(rps))

        # Check if we should scale.
        scale = False
        rps_diff_frac = (abs(rps - self.current_rps)/(self.current_rps+1))
        if rps_diff_frac > .1:
            scale = True
        logger.info("Scale is {}, difference in rps is {}".format(scale,rps_diff_frac))

        self.rps_history.append(rps)
        self.current_rps = rps
        self.req_vector = req_vector
        self.req_index = req_index

        return scale


    def query_model(self):
        ''' Query only based on rps value (not operation) '''
        
        # Get best model context we trained on to use for evluation.
        print("Current RPS: {}".format(self.current_rps))
        train_context_rps = sorted(list(self.rps_map.values()))

        # Get the upper bound from our context.
        query_rps_ub = None
        for context in sorted(train_context_rps):
            if context > self.current_rps:
                query_rps_ub = context
                break

        # Handle rps > all trained context.
        if query_rps_ub is None:
            query_rps_ub = train_context_rps[-1]
            query_rps_lb = train_context_rps[-1]

        # Handle rps == 0.
        elif self.current_rps == 0:
            query_rps_ub = train_context_rps[0]
            query_rps_lb = train_context_rps[0]

        # Handle rps is in our training range.
        else:
            query_rps_lb = train_context_rps[train_context_rps.index(query_rps_ub)-1]

        # Weighting for linear interpolation.
        if query_rps_ub - query_rps_lb == 0:
            w_ub, w_lb = 1, 0
        else:
            w_ub = (self.current_rps - query_rps_lb + 1e-4)/(query_rps_ub - query_rps_lb + 1e-4) 
            w_lb = 1 - w_ub

        # Log models used.
        logger.info("Using UB {} with weight {} and LB {} with weight {}".format(query_rps_ub, w_ub, query_rps_lb, w_lb))

        # Linear interpolate rps.
        rps_interp = ( self.rps_rev_map[query_rps_ub] * w_ub + self.rps_rev_map[query_rps_lb] * w_lb ) / (w_ub + w_lb)

        return rps_interp # Return interpolation info for best model given current rps.


    def get_pod_config(self, num_samples=20000):

        # Currently not autoscaling.
        if self.scaling == False:
            return {}
        
        # Get model for current rps.
        rps_interp = self.query_model()

        # Iterate over all possible choices of training points, query model and select best config.
        start_eval_tp = time.time()
        training_points = [sample_round_multivariate(self.model.bounds) for i in range(num_samples)]

        best_reward, best_training_point = None, None
        for training_point in training_points:
            training_point[0] = rps_interp # Overwrite rps in sample with observed one.
            reward = self.model.predict(training_point)
            if reward == best_reward and sum(training_point) < sum(best_training_point):
                best_reward, best_training_point = reward, training_point

            if best_reward is None or reward > best_reward:
                best_reward, best_training_point = reward, training_point
        end_eval_tp = time.time()
        logger.info("Took {} seconds to eval training points".format(end_eval_tp-start_eval_tp))

        scale_config = {}
        for i, num_replicas in enumerate(best_training_point[1:]):
            service = self.model.search_coords[i]

            # Bound the number of replicas we can have.
            min_replicas, max_replicas = 1, self.app_config.deployments[service]

            # Get the number of replicas to scale to.
            num_replicas = int(math.ceil(num_replicas))
            num_replicas = max(num_replicas, min_replicas)
            num_replicas = min(num_replicas, max_replicas)

            # Record number of replicas for the service.
            scale_config[service] = num_replicas

        logger.info("Current policy: {}".format(scale_config))
        return scale_config
    
        
    def scale_cluster(self, scale_config):

        # Currently not autoscaling.
        if self.scaling == False:
            return
        
        # Compute number of nodes needed (assumes 1->1 matching between pods and nodes).
        num_nodes = sum(scale_config.values()) + self.add_nodes

        # If we are making no changes, return.
        if num_nodes == self.current_nodes and scale_config == self.current_policy:
            logger.info("No scaling action needed, {} nodes.".format(num_nodes))
            return

        # Check if scale up or down event (in terms of nodes).
        scale_up = False
        if num_nodes > self.current_nodes:
            scale_up = True

        # Handle the scale up case by creating nodes and then scaling pods for our services.
        if scale_up is True:
            logger.info("Scaling up from {} to {} nodes.".format(self.current_nodes, num_nodes))

            self.scale_nodes(num_nodes)
            self.scale_pods(scale_config)

        # Handle the scale down case by scaling the deployment down and then decomissioning nodes not needed.
        elif scale_up is False:
            logger.info("Scaling down from {} to {} nodes.".format(self.current_nodes, num_nodes))

            # Scale down pods.
            self.scale_pods(scale_config)
            
            # Get unused nodes, then delete them.
            try:
                unused_nodes, used_nodes = hpa_utils.get_unused_nodes(self.train_config.app_config, wait_time=15)
                if len(unused_nodes) > 0:
                    hpa_utils.delete_unused_nodes(unused_nodes)
                self.current_nodes = num_nodes
            except:
                pass

        return

    def scale_pods(self, scale_config):
        if self.current_policy is None or scale_config != self.current_policy:
            kube_utils.apply_policy(scale_config)
            self.current_policy = scale_config
        return


    def scale_nodes(self, num_nodes):
        if self.current_nodes is None or num_nodes != self.current_nodes:
            # Add nodes for services which are not scaled/controlled by this bandit autoscaler.
            num_nodes += self.add_nodes
            cluster_utils.scale_node_pool(cluster='cola2', project=self.train_config.project_name, zone=self.train_config.zone, node_pool=self.train_config.node_pool, num_nodes=num_nodes)
            self.current_nodes = num_nodes
        return
    
    def turn_on_scaling(self):
        cluster_utils.disable_node_pool_autoscaling(cluster='cola2', 
                                      project=self.train_config.project_name, 
                                      zone=self.train_config.zone, 
                                      node_pool=self.train_config.node_pool,
                                      )
        hpa_utils.delete_autoscaling_policy(config=self.app_config)
        return
    
    def turn_off_scaling(self):
        cluster_utils.enable_node_pool_autoscaling(cluster='cola2', 
                                              project=self.train_config.project_name, 
                                              zone=self.train_config.zone, 
                                              node_pool=self.train_config.node_pool, 
                                              min_nodes=1,
                                              max_nodes=self.train_config.max_nodes
                                              )
        hpa_utils.update_autoscaling_policy(config=self.app_config, cpu_t=50)
        return


if __name__ == "__main__":

    # Parse application from command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config_path', type=str, default='online_boutique')
    parser.add_argument('pod_filter', type=str, default='frontend')
    parser.add_argument('init_nodes', type=int, default=1)
    parser.add_argument('add_nodes', type=int, default=0)

    args = parser.parse_args()
    print(args)

    ca = CherryPickAutoscaler(args.train_config_path, args.pod_filter, args.init_nodes, args.add_nodes)
    ca.run()
