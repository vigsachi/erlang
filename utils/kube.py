import os
import json
import time
import pandas as pd

def scale_deployment(deployment, replicas):
    """
    Scale the number of replicas for a particular deployment.

    Args:
        deployment (str): Name of the deployment
        replicas (int): Number of replicas
    """
    os.system('kubectl scale deployment {} --replicas={}'.format(deployment, replicas))
    return


def apply_policy(policy):
    """
    Scale a set of deployments, given by the policy dict (deployment -> number of replicas).

    Args:
        policy (dict): Mapping between deployments and desired replicas.
    """
    for deployment, replica_value in policy.items():
        os.system('kubectl scale deployment {} --replicas={}'.format(deployment, replica_value))
    return


def get_pod_statistics(cpu_requests=600, mem_requests=2000):
    """
    Get the CPU and Memory utilization for each pod.
    Then convert these values to a % of the CPU and Memory Requests.
    Lastly, group these values by deployment. Returns average CPU and Memory % for each deployment.

    Args:
        cpu_requests (int, optional): CPU requests for each pod. Defaults to 600.
        mem_requests (int, optional): Memory requests for each pod. Defaults to 2000.

    Returns:
        pd.DataFrame: Dataframe with CPU and Memory utilization by deployment.
    """

    # Get raw usage statistics per pod.
    top_pod = os.popen('kubectl top pod').read().split()
    top_df = pd.DataFrame(list(zip(*[iter(top_pod[3:])]*3)))
    top_df[3] = top_df[0].apply(lambda x: '-'.join(x.split('-')[:-2]))
    
    # Compute utilization of the resources.
    top_df[1] = (top_df[1].apply(lambda x: x.split('m')[0]).astype(int) / cpu_requests)*100 # CPU Utilization (%)
    top_df[2] = (top_df[2].apply(lambda x: x.split('M')[0]).astype(int) / mem_requests)*100 # MEM Utilization (%)
    top_df = top_df.groupby([3]).mean().reset_index()

    return top_df


def get_current_deployments():
    """
    Get number of available replicas for each deployment.

    Returns:
        dict: Mapping between deployment -> current number of replicas.
    """
    
    # Get all deployments.
    deployments = json.loads(os.popen('kubectl get deployment -o json').read())
    current_deployments = {}

    # Find the number of available replicas per deployment.
    for dep in range(len(deployments['items'])):
        current_deployments[deployments['items'][dep]['metadata']['name']] = deployments['items'][dep]['status']['readyReplicas']

    if 'istio-ingressgateway' in current_deployments:
        del current_deployments['istio-ingressgateway']

    return current_deployments

def get_current_nodes():
    """
    Get number of available nodes across the application node pool.

    Returns:
        list: Node types used by the application node pool.
    """
    
    # Get all deployments.
    nodes = json.loads(os.popen('kubectl get node -o json').read())
    node_types = [node['metadata']['labels']['beta.kubernetes.io/instance-type'] for node in nodes['items'] if node['metadata']['labels']['cloud.google.com/gke-nodepool'] == 'app-pool']

    return node_types


def measure_num_replicas(fname = 'service_replicas_count.json', num_intervals = 10000000, interval_length = 10):
    """
    Loop which continuously checks the number of replicas. 
    Time to sleep between each check is given by interval_length.

    Args:
        fname (str, optional): File in which to store the number of replicas. Defaults to 'service_replicas_count.json'.
        num_intervals (int, optional): Number of timest to check for replicas. Defaults to 10000000.
        interval_length (int, optional): Time between each check. Defaults to 10.
    """

    # Remove old file.
    os.system('rm {}'.format(fname))

    # Run measurement in loop.
    for i in range(num_intervals):
        try:

            # Get the number of deployments we have running.
            deployments = get_current_deployments()
            total_replicas = sum(deployments.values())
            res_dict = {'services': deployments, 'tot_replicas': total_replicas, 'time': time.time()}

            # Record the deployments.
            with open(fname, 'a+') as obj:
                obj.write(json.dumps(res_dict) + '\n')

            # Wait to try requesting number of deployments until next interval.            
            time.sleep(interval_length)

        except:
            pass
