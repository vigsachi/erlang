from utils.config import Config, EvalConfig, TrainConfig


##############################################
#### Training
##############################################

# 1. 50ms Median Latency
train_cfg = TrainConfig(
                            train_rps=[500, 1000, 1500, 2000], 
                            train_iters=15, 
                            latency_threshold=50, 

                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 

                            locustfile='microservices/hello_world/workloads/default.py',
                            pod_filter='helloworld',
                            req_names=[],
                            search_strategy='random',
                            sample_duration=30,

                            cluster_name='cola-test',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            max_nodes=30,
                            lat_opt='Average Latency'
                        )
