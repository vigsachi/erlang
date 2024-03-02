from utils.config import EvalConfig
from configs.app.hello_world.config import cfg, cfg2, cfg4


##############################################
#### Workloads
##############################################

# 1. Fixed Rate Workload (In Sample)
eval_cfg = EvalConfig(
                            name='fixed_rate_one_core',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test-ob',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[500, 1000, 2000],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=60,
                            num_iters=10,
                            wait_time=120,
                            reset_cluster=False,
                            
                            deployment_path = 'microservices/hello_world/deployments.yaml',
                            gateway_path = 'microservices/hello_world/gateway.yaml',
                            pods_per_node = 1,
                            cluster_type = 'default',
                        )


eval_cfg2 = EvalConfig(
                            name='fixed_rate_two_core',
                            services=cfg2.services, 
                            deployments=cfg2.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test-ob-2',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[500, 1000, 2000],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=60,
                            num_iters=10,
                            wait_time=120,
                            reset_cluster=False,
                            
                            deployment_path = 'microservices/hello_world/deployments.yaml',
                            gateway_path = 'microservices/hello_world/gateway.yaml',
                            pods_per_node = 2,
                            cluster_type = 'default',
                        )

eval_cfg4 = EvalConfig(
                            name='fixed_rate_four_core',
                            services=cfg4.services, 
                            deployments=cfg4.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test-ob-4',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[500, 1000, 2000],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=60,
                            num_iters=10,
                            wait_time=120,
                            reset_cluster=False,
                            
                            deployment_path = 'microservices/hello_world/deployments.yaml',
                            gateway_path = 'microservices/hello_world/gateway.yaml',
                            pods_per_node = 4,
                            cluster_type = 'default',
                        )


# 2. Fixed Rate Workload (In Sample)
eval_cfg_cpu_fr_oos = EvalConfig(
                            name='fixed_rate_oosample',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='http://104.198.190.0/hello',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[1500, 2500],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=60,
                            num_iters=10,
                            wait_time=120,
                            reset_cluster=False
                        )

# 3. Ramp up, ramp down Workload (In Sample)
eval_cfg_ramp = EvalConfig(
                            name='ramp_insample',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='http://104.198.190.0/hello',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[500, 1000, 2000, 1000, 500],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=600,
                            num_iters=1,
                            wait_time=0,
                            reset_cluster=False
                        )

# 4. Ramp up, ramp down Workload (Out of Sample)
eval_cfg_ramp_oos = EvalConfig(
                            name='ramp_oosample',
                            services=cfg.services, 
                            deployments=cfg.deployments,
                            cpu_requests=600,
                            mem_requests=2000,

                            host='http://104.198.190.0/hello',
                            locustfile='microservices/hello_world/workloads/default.py',
                            cluster_name='cola-test',
                            project_name='vig-cloud',
                            zone='us-central1-c',
                            node_pool='app-pool',
                            min_nodes=1,
                            max_nodes=60,

                            application='hello_world',
                            rps_rates=[750, 1250, 1750, 1250, 750],
                            cpu_policies=[10,30,50,70,90],
                            bandit_policy='50_ms',
                            train_config_path='/home/packard2700/cola/models/hello_world/train_test',
                            pod_filter='helloworld',
                            duration=600,
                            num_iters=1,
                            wait_time=0,
                            reset_cluster=False
                        )
