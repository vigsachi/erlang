from utils.config import TrainConfig

##############################################
#### Training
##############################################

# 1. 50ms Median Latency
train_cfg = TrainConfig(
                            train_rps=[100, 200, 300, 400], 
                            train_iters=10, 
                            latency_threshold=50, 

                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 

                            locustfile='microservices/online_boutique/workloads/default.py',
                            pod_filter='frontend',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=45,

                            cluster_name='cola-test-ob',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            max_nodes=130,
                            lat_opt='Average Latency'
                        )

