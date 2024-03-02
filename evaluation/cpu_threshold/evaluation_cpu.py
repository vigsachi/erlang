import os
import sys
sys.path.insert(1, '..')

import time
import math
import pickle
import matplotlib.pyplot as plt

import utils.hpa as hpa_utils
import utils.cluster as cluster_utils
import utils.kube as kube_utils
from utils.locust_loadgen import LoadGenerator
from utils.launch_apps import launch_application, get_host


# Class to run Fixed Rate evaluations.
class FixedRateWorkloadCPU(object):
    def __init__(self, eval_config):

        # Setup configuration
        self.eval_config = eval_config
        self.application = self.eval_config.application
        self.name = self.eval_config.name
        self.rps_rates = self.eval_config.rps_rates
        self.cpu_policies = self.eval_config.cpu_policies
        self.duration = self.eval_config.duration
        self.num_iters = self.eval_config.num_iters
        self.wait_time = self.eval_config.wait_time
        self.reset_cluster = self.eval_config.reset_cluster


        # Setup load generator.
        self.lg = LoadGenerator(host=get_host(self.eval_config.application), 
                                locustfile=self.eval_config.locustfile, 
                                duration=self.eval_config.duration,
                                csv_path=self.eval_config.csv_path,
                                )

        # Reset node pool.
        cluster_utils.scale_node_pool(cluster=self.eval_config.cluster_name, 
                                       project=self.eval_config.project_name, 
                                       zone=self.eval_config.zone, 
                                       node_pool=self.eval_config.node_pool, 
                                       num_nodes=math.ceil(len(self.eval_config.services) / self.eval_config.pods_per_node))

        cluster_utils.enable_node_pool_autoscaling(cluster=self.eval_config.cluster_name, 
                                              project=self.eval_config.project_name, 
                                              zone=self.eval_config.zone, 
                                              node_pool=self.eval_config.node_pool, 
                                              min_nodes=self.eval_config.min_nodes,
                                              max_nodes=self.eval_config.max_nodes
                                              )

        # Launch the application.
        launch_application(config=self.eval_config)
        time.sleep(120)

    def run(self):

        # Evaluated all policies. Run highest threshold first.
        for cpu_policy in self.cpu_policies[::-1]:
            res = self.run_fixed_rate_cpu(cpu_policy)
            self.save_results(res, cpu_policy)
        
        # Clean up.
        hpa_utils.delete_autoscaling_policy(config=self.eval_config)

    def run_fixed_rate_cpu(self, cpu_policy):


        # Authenticate to GKE.
        cluster_utils.authenticate(cluster=self.eval_config.cluster_name)

        # Run each of the RPS values we would like to.
        hpa_utils.update_autoscaling_policy(config=self.eval_config, cpu_t=cpu_policy)
        res = {}
        for rps in self.rps_rates:

            # Measure system response.
            rps_rate_res = self.lg.run_workload(rps_rates = [rps]*self.num_iters)
            res[rps] = rps_rate_res

        return res

    def save_results(self, res, cpu_policy):
    
        # Save results.
        save_path = os.path.join('evaluation', 'results', self.application, self.name, '{}_cpu.pk'.format(cpu_policy))
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

        # Disable cluster autoscaling.    
        cluster_utils.disable_node_pool_autoscaling(cluster=self.eval_config.cluster_name, 
                                                    project=self.eval_config.project_name, 
                                                    zone=self.eval_config.zone, 
                                                    node_pool=self.eval_config.node_pool)

        # Allocate one node per service.
        cluster_utils.scale_node_pool(cluster=self.eval_config.cluster_name, 
                                        project=self.eval_config.project_name, 
                                        zone=self.eval_config.zone, 
                                        node_pool=self.eval_config.node_pool, 
                                        num_nodes=len(self.eval_config.services))

        # Enable cluster autoscaling.    
        cluster_utils.enable_node_pool_autoscaling(cluster=self.eval_config.cluster_name, 
                                                project=self.eval_config.project_name, 
                                                zone=self.eval_config.zone, 
                                                node_pool=self.eval_config.node_pool, 
                                                min_nodes=self.eval_config.min_nodes,
                                                max_nodes=self.eval_config.max_nodes
                                                )

        return
