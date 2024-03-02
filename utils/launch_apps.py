import os
import sys
import time
sys.path.insert(1, '..')

import argparse

APPLICATIONS = ['hello_world', 'book_info', 'online_boutique', 'hotel_reservation', 'sock_shop', 'train_ticket']
SUFFIXES = {
            'online_boutique': ':80', 
            'book_info': '/productpage', 
            'hello_world': ':80/hello', 
            'hotel_reservation': ':80', 
            'sock_shop':':80',
            'train_ticket': ':80',
           }

def launch_application(config, delete_existing_apps=True):
    """
    Launch the specified microservice application.
    Requires cloud provider authentication to already be configured.

    Args:
        app_name (str, optional): Name of the application. Defaults to 'online_boutique'.

    Returns:
        _type_: Landing page URL for the application.
    """
    
    # Stop all applications
    if delete_existing_apps is True:
        for application in APPLICATIONS:
            os.system('kubectl delete -f microservices/{}/deployments.yaml'.format(application))
            os.system('kubectl delete -f microservices/{}/gateway.yaml'.format(application))


    # Launch specified application
    os.system('kubectl apply -f {}'.format(config.deployment_path))
    os.system('kubectl apply -f {}'.format(config.gateway_path))


    # Get host of the application and return.
    time.sleep(30) # Wait for the app and gateway to come up.
    ingress_host = os.popen("kubectl -n default get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'").read()

    # Create users for train ticket application.
    if config.application == 'trainticket':
        time.sleep(60)
        os.system('locust -f load_generator/locustfiles/trainticket/create_users.py --headless -u 100 -r 10 --host http://{ingress_host} --run-time 240s'.format(ingress_host=ingress_host))
        os.system('locust -f load_generator/locustfiles/trainticket/create_users.py --headless -u 100 -r 10 --host http://{ingress_host} --run-time 240s'.format(ingress_host=ingress_host))

    print("Host = {ingress_host}".format(ingress_host=ingress_host+SUFFIXES[config.application]))
    return

def delete_applications():
    """
    Deletes the deployment and gateway for all applications.
    """

    # Stop all applications
    for application in APPLICATIONS:
        os.system('kubectl delete -f microservices/{}/deployments.yaml'.format(application))
        os.system('kubectl delete -f microservices/{}/gateway.yaml'.format(application))

    return

def get_host(app_name):
    """
    Get the host URL for an application.

    Args:
        app_name (str): Name of application

    Returns:
        str: host URL
    """
    ingress_host = os.popen("kubectl -n default get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}'").read()
    return 'http://' + ingress_host+SUFFIXES[app_name]
