from utils.config import Config

##############################################
#### Application
##############################################

# Online Boutique Application
cfg = Config(
                name='online_boutique',
                application='online_boutique',
                services=[
                            'frontend', 
                            'productcatalogservice', 
                            'currencyservice', 
                            'recommendationservice', 
                            'cartservice', 
                            'adservice', 
                            'checkoutservice', 
                            'shippingservice', 
                            'redis', 
                            'paymentservice', 
                            'emailservice'
                        ],
                deployments={
                                'frontend':30, 
                                'productcatalogservice':10, 
                                'currencyservice':10, 
                                'recommendationservice':10, 
                                'cartservice':10, 
                                'adservice':10, 
                                'checkoutservice':10, 
                                'shippingservice':10, 
                                'redis':10, 
                                'paymentservice':10, 
                                'emailservice':10
                            },
                cpu_requests=500,
                mem_requests=1900,

                host='',
                autoscale_path='',
                cluster_name='cola-test-ob',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/online_boutique/deployments.yaml',
                gateway_path = 'microservices/online_boutique/gateway.yaml',
                pods_per_node = 1,
                cluster_type = 'default',
            )


cfg2 = Config(
                name='online_boutique',
                application='online_boutique',
                services=[
                            'frontend', 
                            'productcatalogservice', 
                            'currencyservice', 
                            'recommendationservice', 
                            'cartservice', 
                            'adservice', 
                            'checkoutservice', 
                            'shippingservice', 
                            'redis', 
                            'paymentservice', 
                            'emailservice'
                        ],
                deployments={
                                'frontend':30, 
                                'productcatalogservice':10, 
                                'currencyservice':10, 
                                'recommendationservice':10, 
                                'cartservice':10, 
                                'adservice':10, 
                                'checkoutservice':10, 
                                'shippingservice':10, 
                                'redis':10, 
                                'paymentservice':10, 
                                'emailservice':10
                            },
                cpu_requests=500,
                mem_requests=1900,

                host='',
                autoscale_path='',
                cluster_name='cola-test-ob-2',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/online_boutique/deployments.yaml',
                gateway_path = 'microservices/online_boutique/gateway.yaml',
                pods_per_node = 2,
                cluster_type = 'default',
            )


cfg4 = Config(
                name='online_boutique',
                application='online_boutique',
                services=[
                            'frontend', 
                            'productcatalogservice', 
                            'currencyservice', 
                            'recommendationservice', 
                            'cartservice', 
                            'adservice', 
                            'checkoutservice', 
                            'shippingservice', 
                            'redis', 
                            'paymentservice', 
                            'emailservice'
                        ],
                deployments={
                                'frontend':30, 
                                'productcatalogservice':10, 
                                'currencyservice':10, 
                                'recommendationservice':10, 
                                'cartservice':10, 
                                'adservice':10, 
                                'checkoutservice':10, 
                                'shippingservice':10, 
                                'redis':10, 
                                'paymentservice':10, 
                                'emailservice':10
                            },
                cpu_requests=500,
                mem_requests=1900,

                host='',
                autoscale_path='',
                cluster_name='cola-test-ob-4',
                project_name='vig-cloud',
                zone='us-central1-c',

                deployment_path = 'microservices/online_boutique/deployments.yaml',
                gateway_path = 'microservices/online_boutique/gateway.yaml',
                pods_per_node = 4,
                cluster_type = 'default',
            )

