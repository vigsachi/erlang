import os
import sys
sys.path.insert(1, '..')

import time
import pickle
import matplotlib.pyplot as plt

import utils.hpa as hpa_utils
import utils.cluster as cluster_utils
from utils.locust_loadgen import LoadGenerator
from utils.launch_apps import launch_application

    
# Class to run Fixed Rate evaluations.
class FixedRateWorkloadMEM(object):
    def __init__(self, eval_config):

        # Setup configuration
        self.eval_config = eval_config
        self.application = self.eval_config.application
        self.name = self.eval_config.name
        self.rps_rates = self.eval_config.rps_rates
        self.mem_policies = self.eval_config.cpu_policies
        self.duration = self.eval_config.duration
        self.num_iters = self.eval_config.num_iters
        self.wait_time = self.eval_config.wait_time
        self.reset_cluster = self.eval_config.reset_cluster

        # Launch the application.
        launch_application(config=self.eval_config)

        # Setup load generator.
        self.lg = LoadGenerator(host=self.eval_config.host, 
                                locustfile=self.eval_config.locustfile, 
                                duration=self.eval_config.duration,
                                csv_path=self.eval_config.csv_path,
                                )

        # Reset node pool.
        cluster_utils.scale_node_pool(cluster=self.eval_config.cluster_name, 
                                       project=self.eval_config.project_name, 
                                       zone=self.eval_config.zone, 
                                       node_pool=self.eval_config.node_pool, 
                                       num_nodes=len(self.eval_config.services))

        cluster_utils.enable_node_pool_autoscaling(cluster=self.eval_config.cluster_name, 
                                              project=self.eval_config.project_name, 
                                              zone=self.eval_config.zone, 
                                              node_pool=self.eval_config.node_pool, 
                                              min_nodes=self.eval_config.min_nodes,
                                              max_nodes=self.eval_config.max_nodes
                                              )
        time.sleep(120)

    def run(self):

        # Evaluated all policies. Run highest threshold first.
        for mem_policy in self.mem_policies[::-1]:
            res = self.run_fixed_rate_mem(mem_policy)
            self.save_results(res, mem_policy)
        
        # Clean up.
        hpa_utils.delete_autoscaling_policy(config=self.eval_config)

    def run_fixed_rate_mem(self, mem_policy):

        # Authenticate to GKE.
        cluster_utils.authenticate(cluster=self.eval_config.cluster_name)

        # Update autoscaling policy
        hpa_utils.update_autoscaling_policy_mem(config=self.eval_config, mem_t=mem_policy)

        # Run each of the RPS values we would like to.
        res = {}
        for rps in self.rps_rates:

            # Measure system response.
            rps_rate_res = self.lg.run_workload(rps_rates = [rps]*self.num_iters)
            res[rps] = rps_rate_res
        
        return res

    def save_results(self, res, mem_policy):
        
        # Save results.
        save_path = os.path.join('evaluation', 'results', self.application, self.name, '{}_mem.pk'.format(mem_policy))
        if not os.path.exists('evaluation'):
            os.mkdir('evaluation')
        if not os.path.exists(os.path.join('evaluation', 'results')):
            os.mkdir(os.path.join('evaluation', 'results'))
        if not os.path.exists(os.path.join('evaluation', 'results', self.application)):
            os.mkdir( os.path.join('evaluation', 'results', self.application ))
        if not os.path.exists(os.path.join('evaluation', 'results', self.application, self.name)):
            os.mkdir( os.path.join('evaluation', 'results', self.application, self.name))
        pickle.dump(res, open(save_path, "wb"))

        return

    def reset_node_pool(self):
    
        # Allocate one node per service.
        cluster_utils.scale_node_pool(cluster=self.eval_config.cluster_name, 
                                        project=self.eval_config.project_name, 
                                        zone=self.eval_config.zone, 
                                        node_pool=self.eval_config.node_pool, 
                                        num_nodes=len(self.eval_config.services))

        return
