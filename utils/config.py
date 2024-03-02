import copy

class Config(object):
    def __init__(self, name, application, services, deployments, cpu_requests, mem_requests, host, cluster_name, 
                       project_name, zone, autoscale_path='', deployment_path='', gateway_path='', 
                       pods_per_node=1, cluster_type='default'):
        """
        Configuration of a microservice application.
        Includes information on services, node ranges and endpoint.

        Args:
            name (str): Name of the deployed application.
            services (list): Services we can consider scaling.
            deployments (dict): Max scaling value accompanying each service.
            cpu_requests (int): Requested cpu for each pod.
            mem_requests (int): Requested memory for each pod.
            host (_type_): Deployed application endpoint.
            autoscale_path (str, optional): Path for custom autoscalers. Defaults to ''.
        """
        self.name = name
        self.application = application
        self.services = services
        self.deployments = deployments
        self.cpu_requests = cpu_requests
        self.mem_requests = mem_requests
        self.host = host
        self.autoscale_path = autoscale_path
        self.cluster_name = cluster_name
        self.project_name = project_name
        self.zone = zone
        self.deployment_path = deployment_path
        self.gateway_path = gateway_path
        self.pods_per_node = pods_per_node
        self.cluster_type = cluster_type

class TrainConfig(object):
    def __init__(self, train_rps, 
                       train_iters,
                       latency_threshold, 
                       c,
                       w_l, 
                       w_i, 
                       min_iters, 
                       locustfile,
                       project_name,
                       cluster_name,
                       zone,
                       node_pool,
                       req_names,
                       search_strategy='cpu',
                       sample_duration=25,
                       max_nodes=130,
                       lat_opt='Average Latency',
                       pod_filter='frontend'):
        """
        Training configuration for learning autoscaling policy.

        Args:
            train_rps (list): List of RPS values that we will train a model for.
            train_iters (int): Maximum number of training iterations before quitting.
            latency_threshold (int): Desired latency threshold for application.
            c (int): Exploration parameter for bandit.
            w_l (int): Weight for latency objective.
            w_i (int): Weight for cost objective
            min_iters (int): Minimum number of iterations for training.
            locustfile (str): Path to the workload (as locustfile) to be evaluated.
            project_name (str): Name of the project which hosts the cluster. Defaults to 'vig-cloud'.
            cluster (str): Name of the GKE cluster. Defaults to 'cola2'.
            zone (str): Zone in which cluster is located. Defaults to 'us-central1-c'.
            node_pool (str): Name of the node pool to scale. Defaults to 'ob-pool'.
            req_names (list): Manual override of operations that should be considered as part of the context.
            search_strategy (str, optional): How to select pods for autoscaling (e.g. 'cpu' percent utilization). Defaults to 'cpu'.
            sample_duration (int, optional): How long of a duration for selecting samples. Defaults to 25.
            max_nodes (int, optional): Maximum nodes in node pool for training. Defaults to 130.
            lat_opt (str, optional): Type of latency to optimize (e.g. median, 90%ile). Defaults to 'Average Latency'.
            pod_filter (str): Frontend pod of the application.
        """

        # List of context to train on.
        self.train_rps = sorted(train_rps)

        # Training objectives and parameters.
        self.train_iters = train_iters
        self.latency_threshold = latency_threshold
        self.c = c
        self.w_l = w_l
        self.w_i = w_i
        self.min_iters = min_iters
        self.search_strategy = search_strategy
        self.sample_duration = sample_duration
        self.locustfile = locustfile

        # Cluster configs
        self.cluster_name = cluster_name
        self.project_name = project_name
        self.zone = zone
        self.node_pool = node_pool
        self.max_nodes = max_nodes

        # Warm start configs.
        self.warm_start = False
        self.cpu_t = 50
        self.cpu_duration = 400
        self.num_workers = 10
        self.csv_path = 'logs/scratch/cola_lg'
        self.lat_opt = lat_opt

        # A place to save configs and models.
        self.config_map = {}
        self.model_map = {}

        # Inference params.
        self.req_names = req_names
        self.pod_filter = pod_filter

    def add_context(self, context, config, model):

        # Save most recent config and model to the train config.
        if config:
            self.config_map[context] = copy.deepcopy(config)
        if model:
            self.model_map[context] = copy.deepcopy(model)
        
        return


class EvalConfig(object):
    def __init__(self, name, services, deployments, cpu_requests, mem_requests, 
                host, locustfile, cluster_name, project_name, zone, node_pool, min_nodes, max_nodes,
                application, rps_rates, cpu_policies, bandit_policy, pod_filter, train_config_path, duration, 
                num_iters, wait_time, reset_cluster, autoscale_path='', second_context=None,
                deployment_path='', gateway_path='', pods_per_node=1, cluster_type='default'):
        """
        Evaluation configuration for learned autoscaling policy.

        Args:
            name (str): Name of the deployed application.
            services (list): Services we can consider scaling.
            deployments (dict): Max scaling value accompanying each service.
            cpu_requests (int): Requested cpu for each pod.
            mem_requests (int): Requested memory for each pod.
            host (str): Deployed application endpoint.
            locustfile (str): Path to the workload (as locustfile) to be evaluated.
            cluster (str): Name of the GKE cluster. Defaults to 'cola2'.
            project (str): Name of the project which hosts the cluster. Defaults to 'vig-cloud'.
            zone (str): Zone in which cluster is located. Defaults to 'us-central1-c'.
            node_pool (str): Name of the node pool to scale. Defaults to 'ob-pool'.
            min_nodes (int): Minimum nodes in cluster. Defaults to 1.
            max_nodes (int): Maximum nodes in cluster. Defaults to 10.
            application (str): Name of the application.
            rps_rates (list): RPS values to evaluate.
            cpu_policies (list): CPU autoscaling thresholds to evaluate.
            bandit_policy (str): Name of the bandit policy.
            pod_filter (str): Frontend pod of the application.
            train_config_path (str): Path to the trained model.
            duration (str): How long to run each RPS value in evaluation.
            num_iters (int): Number of times to run each duration for each RPS value.
            wait_time (int): Time to wait for cluster to setup.
            reset_cluster (bool): Whether or not to delete all existing nodes and pods in cluster.
            autoscale_path (str, optional): Path for custom autoscaler yaml (e.g. memory). Defaults to ''.
        """

        # Name of the deployed application.
        self.name = name

        # Services we can consider scaling and the scaling range accompanying each of them.
        self.services = services
        self.deployments = deployments

        # Requested cpu and memory for each pod.
        self.cpu_requests = cpu_requests
        self.mem_requests = mem_requests

        # Deployed application endpoint.
        self.host = host
        self.locustfile = locustfile
        self.csv_path = 'logs/scratch/cola_lg'
        self.cluster_name = cluster_name
        self.project_name = project_name
        self.zone = zone
        self.node_pool = node_pool
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.autoscale_path = autoscale_path
        self.second_context = second_context

        # Eval settings.
        self.application = application
        self.rps_rates = rps_rates
        self.cpu_policies = cpu_policies
        self.bandit_policy = bandit_policy
        self.train_config_path = train_config_path
        self.pod_filter = pod_filter
        self.duration = duration
        self.num_iters = num_iters
        self.wait_time = wait_time
        self.reset_cluster = reset_cluster

        # Application deployment settings
        self.deployment_path = deployment_path
        self.gateway_path = gateway_path
        self.pods_per_node = pods_per_node
        self.cluster_type = cluster_type
