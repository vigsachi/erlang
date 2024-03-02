from utils.config import Config

##############################################
#### Application Config
##############################################

# Book Info Application
cfg = Config(
                name='book_info',
                application='book_info',
                services=[
                            'details', 
                            'productpage', 
                            'ratings', 
                            'reviews', 
                        ],
                deployments={
                                'details': 20,
                                'productpage': 40, 
                                'ratings': 20,
                                'reviews': 20,
                            },
                cpu_requests=500,
                mem_requests=1900,
                host='',
                autoscale_path='',
                cluster_name='cola-test-bi',
                project_name='vig-cloud',
                zone='us-central1-c',
                
                deployment_path = 'microservices/book_info/deployments.yaml',
                gateway_path = 'microservices/book_info/gateway.yaml',
                pods_per_node = 1,
                cluster_type = 'default',
            )


cfg2 = Config(
                name='book_info',
                application='book_info',
                services=[
                            'details', 
                            'productpage', 
                            'ratings', 
                            'reviews', 
                        ],
                deployments={
                                'details': 20,
                                'productpage': 40, 
                                'ratings': 20,
                                'reviews': 20,
                            },
                cpu_requests=500,
                mem_requests=1900,
                host='',
                autoscale_path='',
                cluster_name='cola-test-bi-2',
                project_name='vig-cloud',
                zone='us-central1-c',
                
                deployment_path = 'microservices/book_info/deployments.yaml',
                gateway_path = 'microservices/book_info/gateway.yaml',
                pods_per_node = 2,
                cluster_type = 'default',
            )

cfg4 = Config(
                name='book_info',
                application='book_info',
                services=[
                            'details', 
                            'productpage', 
                            'ratings', 
                            'reviews', 
                        ],
                deployments={
                                'details': 20,
                                'productpage': 40, 
                                'ratings': 20,
                                'reviews': 20,
                            },
                cpu_requests=500,
                mem_requests=1900,
                host='',
                autoscale_path='',
                cluster_name='cola-test-bi-4',
                project_name='vig-cloud',
                zone='us-central1-c',
                
                deployment_path = 'microservices/book_info/deployments.yaml',
                gateway_path = 'microservices/book_info/gateway.yaml',
                pods_per_node = 4,
                cluster_type = 'default',
            )