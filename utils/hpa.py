import os
import json
import time

def update_autoscaling_policy(config, cpu_t=5):
    """
    Update CPU autoscaling thresholds for all microservice deployments.

    Args:
        config (utils.config.Config): Configuration for the application.
        cpu_t (int, optional): CPU autoscaling threshold to apply. Defaults to 5.
    """

    # Iterate through each deployment and apply cpu autoscaling policies.
    for deployment, replicas in config.deployments.items():
                
        # Delete old autoscaling policy.
        os.system('kubectl delete hpa {deployment}'.format(deployment=deployment))
        os.system('kubectl delete hpa {deployment}-mem'.format(deployment=deployment))

        # Scale deployment to 1 replica.
        os.system('kubectl scale deployment {deployment} --replicas=1'.format(deployment=deployment))

        # Set new autoscaling policy on deployment.
        os.system('kubectl autoscale deployment {deployment} --max {max_nodes} --min {min_nodes} --cpu-percent {cpu_t}'.format(
                        cpu_t=cpu_t, max_nodes=replicas, min_nodes=1, deployment=deployment))

    return

def update_autoscaling_policy_mem(config, mem_t=5):
    """
    Update Memory threshold autoscaling thresholds for all microservice deployments.

    Args:
        config (utils.config.Config): Configuration for the application.
        mem_t (int, optional): CPU autoscaling threshold to apply. Defaults to 5.
    """

    # Iterate through each deployment and apply cpu autoscaling policies.
    for deployment, replicas in config.deployments.items():
                
        # Delete old autoscaling policy.
        os.system('kubectl delete hpa {deployment}-mem'.format(deployment=deployment))
        
        # Scale deployment to 1 replica.
        os.system('kubectl scale deployment {deployment} --replicas=1'.format(deployment=deployment))

    # Set new autoscaling policy on deployment. Note number of replicas is set within yaml file for mem autoscalers.
    os.system('kubectl apply -f {autoscale_path}/mem_autoscale_{mem_t}.yaml'.format(
                    mem_t=mem_t, autoscale_path=config.autoscale_path))

    return


def delete_autoscaling_policy(config):
    """
    Delete autoscaling policies for all microservices.

    Args:
        config (utils.config.Config): Configuration for the application.
    """
    for deployment in config.deployments:
        os.system('kubectl delete hpa {deployment}'.format(deployment=deployment))
    return


def get_unused_nodes(config, node_filter='ob-pool', wait_time=0):
    """
    Gets list of nodes (VMs) which are not being used by any application pod.

    Args:
        config (utils.config.Config): Configuration for the application.
        node_filter (str, optional): Filter for the nodes on which applications are located. Defaults to 'ob-pool'.
        wait_time (int, optional): _description_. Defaults to 0.

    Returns:
        list, list: Lists of nodes without and with application pods respectively. 
    """

    time.sleep(wait_time)

    # Get pods and nodes for the cluster.
    pods = json.loads(os.popen('kubectl get pods -o json').read())
    nodes = json.loads(os.popen('kubectl get nodes -o json').read())

    # List of node names.
    node_names = [nodes['items'][i]['metadata']['name'] for i in range(len(nodes['items']))]
    node_names = [node for node in node_names if node_filter in node]

    # List of nodes occupied by our pods
    pod_nodes = [pods['items'][i]['spec']['nodeName'] for i in range(len(pods['items']))]

    # Get nodes unoccupied with deployments.
    nodes_to_delete = set(node_names).difference(set(pod_nodes))

    return nodes_to_delete, pod_nodes

def delete_unused_nodes(unused_nodes, zone='us-central1-c'):
    """
    Deletes the nodes not being used by our application's pods.
    Used, for example, when scaling down the number of pods in our cluster.

    Args:
        unused_nodes (list): List of nodes to remove from our cluster.
        zone (str, optional): Zone on which to delete nodes. Defaults to 'us-central1-c'.
    """

    # Drain the nodes to be deleted and cordon them.
    for node in unused_nodes:
        try:
           os.system('kubectl drain {node} --force'.format(node=node))
        except:
            pass

    # Delete the cordoned nodes.
    group_id = os.popen("gcloud container clusters describe cola --zone us-central1-c --format json | jq  --raw-output '.instanceGroupUrls[1]' | rev | cut -d'/' -f 1 | rev").read().split('\n')[0]
    os.system('gcloud compute instance-groups managed delete-instances {group_id} --zone {zone} --instances={instances}'.format(zone=zone, group_id=group_id, instances=','.join(unused_nodes)))
    return
