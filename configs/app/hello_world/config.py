from utils.config import Config, EvalConfig, TrainConfig


##############################################
#### Application
##############################################

# Hello World Application
cfg = Config(
                name='hello_world',
                application='hello_world',
                services=[
                            'helloworld', 
                        ],
                deployments={
                                'helloworld': 30,
                            },
                cpu_requests=600,
                mem_requests=2000,
                host='',
                autoscale_path='',
                cluster_name='cola-test-ob',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/hello_world/deployments.yaml',
                gateway_path = 'microservices/hello_world/gateway.yaml',
                pods_per_node = 1,
                cluster_type = 'default',
           )


cfg2 = Config(
                name='hello_world',
                application='hello_world',
                services=[
                            'helloworld', 
                        ],
                deployments={
                                'helloworld': 30,
                            },
                cpu_requests=600,
                mem_requests=2000,
                host='',
                autoscale_path='',
                cluster_name='cola-test-ob-2',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/hello_world/deployments.yaml',
                gateway_path = 'microservices/hello_world/gateway.yaml',
                pods_per_node = 2,
                cluster_type = 'default',
           )

cfg4 = Config(
                name='hello_world',
                application='hello_world',
                services=[
                            'helloworld', 
                        ],
                deployments={
                                'helloworld': 30,
                            },
                cpu_requests=600,
                mem_requests=2000,
                host='',
                autoscale_path='',
                cluster_name='cola-test-ob-4',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/hello_world/deployments.yaml',
                gateway_path = 'microservices/hello_world/gateway.yaml',
                pods_per_node = 4,
                cluster_type = 'default',
           )
