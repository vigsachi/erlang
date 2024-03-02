from utils.config import Config, EvalConfig, TrainConfig

################################################## 
# Application
##############################################

# Book Info Application
cfg = Config(
                name='sock_shop',
                services=[
                            'carts', 
                            #'carts-db',
                            'catalogue',
                            #'catalogue-db',
                            'front-end',
                            'orders',
                            #'orders-db',
                            'payment',
                            'queue-master',
                            'rabbitmq',
                            #'session-db',
                            'shipping',
                            'user',
                            #'user-db',
                        ],
                deployments={
                            'carts': 10, 
                            #'carts-db': 10,
                            'catalogue': 10,
                            #'catalogue-db': 10,
                            'front-end': 10,
                            'orders': 10,
                            #'orders-db': 10,
                            'payment': 10,
                            'queue-master': 10,
                            'rabbitmq': 10,
                            #'session-db': 10,
                            'shipping': 10,
                            'user': 10,
                            #'user-db': 10,
                            },
                cpu_requests=600,
                mem_requests=2000,
                host='http://35.232.246.231:80',
                autoscale_path='microservices/ss/mem_autoscale'
            )


##############################################
#### Training
##############################################

# 1. 50ms Median Latency
train_cfg = TrainConfig(
                            train_rps=[200,300,400,500], 
                            train_iters=15, 
                            latency_threshold=50, 
                            c=2, 
                            w_l=5, 
                            w_i=15, 
                            min_iters=5, 
                            locustfile='load_generator/locustfiles/sockshop/default.py',
                            pod_filter='front-end',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=80,
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            max_nodes=100,
                            lat_opt='Average Latency'
                        )



train_cfg_tail = TrainConfig(
                            train_rps=[200,300,400,500],
                            train_iters=15,
                            latency_threshold=100,
                            c=2,
                            w_l=5,
                            w_i=15,
                            min_iters=5,
                            locustfile='load_generator/locustfiles/sockshop/default.py',
                            pod_filter='front-end',
                            req_names=[],
                            search_strategy='cpu',
                            sample_duration=80,
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            max_nodes=100,
                            lat_opt='90p Latency'
                        )


eval_cfg_ramp = EvalConfig(
                            name='ramp_insample',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host=cfg.host,
                            locustfile='load_generator/locustfiles/sockshop/default.py',
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            min_nodes=1,
                            max_nodes=200,

                            application='sock_shop',
                            rps_rates=[200,300,500,300,200],
                            cpu_policies=[30,70],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/autoscale-bandit/models/sock_shop/bandit-50-large',
                            pod_filter='front-end',
                            duration=600,
                            num_iters=1,
                            wait_time=0,
                            reset_cluster=False,
                            autoscale_path=cfg.autoscale_path,
                        )


eval_cfg_ramp_oos = EvalConfig(
                            name='ramp_oosample',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host=cfg.host,
                            locustfile='load_generator/locustfiles/sockshop/default.py',
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            min_nodes=1,
                            max_nodes=200,

                            application='sock_shop',
                            rps_rates=[250,350,400,350,250],
                            cpu_policies=[30,70],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/autoscale-bandit/models/sock_shop/bandit-50-large',
                            pod_filter='front-end',
                            duration=600,
                            num_iters=1,
                            wait_time=0,
                            reset_cluster=False,
                            autoscale_path=cfg.autoscale_path,
                        )


eval_cfg_ramp_hilo = EvalConfig(
                            name='ramp_hilo',
                            services=cfg.services,
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host=cfg.host,
                            locustfile='load_generator/locustfiles/sockshop/default.py',
                            cluster_name='cola2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='ob-pool',
                            min_nodes=1,
                            max_nodes=200,

                            application='sock_shop',
                            rps_rates=[232,288,237,391],
                            cpu_policies=[30,70],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/autoscale-bandit/models/sock_shop/bandit-50-large',
                            pod_filter='front-end',
                            duration=600,
                            num_iters=1,
                            wait_time=0,
                            reset_cluster=False,
                            autoscale_path=cfg.autoscale_path,
                        )
