import os
import sys
sys.path.insert(1, '..')

import time
import pickle
#import matplotlib.pyplot as plt

import utils.hpa as hpa_utils
import utils.cluster as cluster_utils
from utils.locust_loadgen import LoadGenerator
from utils.launch_apps import launch_application, get_host
    

class FixedRateWorkloadBandit(object):
    def __init__(self, eval_config, generalize_dist=False, launch_app=False):

        # Setup configuration
        self.eval_config = eval_config
        self.application = self.eval_config.application
        self.name = self.eval_config.name
        self.rps_rates = self.eval_config.rps_rates
        self.duration = self.eval_config.duration
        self.num_iters = self.eval_config.num_iters
        self.wait_time = self.eval_config.wait_time
        self.reset_cluster = self.eval_config.reset_cluster
        self.train_config_path = self.eval_config.train_config_path
        self.bandit_policy = self.eval_config.bandit_policy

        # Delete HPA for the application.
        hpa_utils.delete_autoscaling_policy(config=self.eval_config)

        # Setup load generator.
        self.lg = LoadGenerator(host=get_host(self.eval_config.application), 
                                locustfile=self.eval_config.locustfile, 
                                duration=self.eval_config.duration,
                                csv_path=self.eval_config.csv_path,
                                )

        # Disable cluster autoscaling and set nodes = number of services.  
        cluster_utils.disable_node_pool_autoscaling(cluster=self.eval_config.cluster_name, 
                                                  project=self.eval_config.project_name, 
                                                  zone=self.eval_config.zone, 
                                                  node_pool=self.eval_config.node_pool)

        self.reset_node_pool()
        launch_application(config=self.eval_config)
        time.sleep(120)

        # Setup bandit autoscaler and run autoscaler in background process.
        cmd = 'python3 inference/cola/inference_cola.py {} {} {} 0 &'.format(self.train_config_path, self.eval_config.pod_filter, len(self.eval_config.services))
        print(cmd)
        os.system(cmd)


    def run(self):

        # Evaluated all policies.
        res = self.run_fixed_rate()
        self.save_results(res)

        # Stop the bandit autoscaling process.
        os.system('pkill -f inference_cola')

    def run_fixed_rate(self):

        # Authenticate to GKE.
        cluster_utils.authenticate(cluster=self.eval_config.cluster_name)

        # Run each of the RPS values we would like to.
        res = {}
        for rps in self.rps_rates:

            # Measure system response.
            rps_rate_res = self.lg.run_workload(rps_rates = [rps]*self.num_iters)
            res[rps] = rps_rate_res
        
        return res

    def save_results(self, res):

        # Save results.
        save_path = os.path.join('evaluation', 'results', self.application, self.name, '{}_cola.pk'.format(self.bandit_policy))
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
