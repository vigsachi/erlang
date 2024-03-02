from utils.config import TrainConfig


##############################################
#### Training
##############################################

# 1. 50ms Median Latency
train_cfg = TrainConfig(
                            train_rps=[200,400,600,800], 
                            train_iters=15, 
                            latency_threshold=50, 

                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 

                            locustfile='microservices/book_info/workloads/default.py',
                            pod_filter='productpage',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=25,

                            cluster_name='cola-test-bi',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            max_nodes=60,
                            lat_opt='Average Latency'
                        )



train_cfg2 = TrainConfig(
                            train_rps=[200,400,600,800], 
                            train_iters=15, 
                            latency_threshold=50, 

                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 

                            locustfile='microservices/book_info/workloads/default.py',
                            pod_filter='productpage',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=25,

                            cluster_name='cola-test-bi-2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            max_nodes=60,
                            lat_opt='Average Latency'
                        )

train_cfg4 = TrainConfig(
                            train_rps=[200,400,600,800], 
                            train_iters=15, 
                            latency_threshold=50, 

                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 

                            locustfile='microservices/book_info/workloads/default.py',
                            pod_filter='productpage',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=25,

                            cluster_name='cola-test-bi-4',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            max_nodes=60,
                            lat_opt='Average Latency'
                        )



# 2. 50ms Median Latency (Large range of requests)
train_cfg_ldr = TrainConfig(
                            train_rps=[50,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
                            train_iters=15,
                            latency_threshold=50,
                            c=2,
                            w_l=5,
                            w_i=15,
                            min_iters=5,
                            locustfile='load_generator/locustfiles/bookinfo/default.py',
                            pod_filter='productpage',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=60,
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            max_nodes=100,
                            lat_opt='Average Latency'
                        )
