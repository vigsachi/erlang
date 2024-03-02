# COLA - Learned Autoscalers for Kubernetes

COLA is a python package that trains and runs learned **autoscalers** for Kubernetes. These autoscalers differ from those built in to Kubernetes in a couple of key ways:

- COLA optimizes for latency and cost, not machine utilization.
  - COLA is trained to optimize the number of microservice replicas in your cluster to satisfy a latency constraint which you provide (e.g. a median end to end latency of 50ms) at the lowest cost across a set of application workloads.
- COLA adapts its policy to your application's workload.
  - COLA, once trained, runs in inference mode where it observes your applications current workload and applies an optimized autoscaling policy.

<!-- Typical results and link to paper -->
On average, we've seen that COLA reduces the cluster cost needed to meet a latency target by roughly 30% over built in Kubernetes autoscaling policies. The full details and experimentation results can be accessed here: https://arxiv.org/abs/2112.14845.

# Google Kubernetes Engine (GKE) Workflow:

To train and evaluate COLA on some sample applications, we will perform the following steps:
1. Create a Kubernetes cluster on GKE
2. Train COLA on a latency target and set of application workloads.
3. Evaluate COLA on another set of application workloads.


## 0. Installation

First, clone this repository:

    git clone https://github.com/vig-sachi/cola.git

Then, install package dependencies for your operating system and python3 environment.

### Debian
    python3 dependencies/install_deb.py

The full list of packages installed can be viewed in **`dependencies/os/deb/install_req.sh`**.

### Mac
    python3 dependencies/install_mac.py

The full list of packages installed can be viewed in **`dependencies/python/requirements.txt`**.


## 1. Creating a Kubernetes Cluster


Install dependencies on a Google Compute Engine VM of your choice. 

Then, install an authentication key for Google Cloud Monitoring which we'll use to log, monitor and query application workloads. Instructions are given at this link: https://cloud.google.com/monitoring/docs/reference/libraries#setting_up_authentication. Once a monitoring json is created, run the following command in your terminal shell:

    export GOOGLE_APPLICATION_CREDENTIALS="path/to/file.json"

The following example script creates a cluster for the `Hello World` application and launches the application.

    python3 examples/hello_world/hello_world_setup.py

The script does the following:
1. Authenticate your VM's command line interface to Google Cloud and set the Project ID, Zone and cluster name as specified in **`configs/app/hello_world/config.py`**.
2. Create two node pools within the cluster -- one for your applications pods and the other for Istio pods (including an ingress gateway).
3. Install Helm, Istio, and the Istio Stackdriver plugins on your cluster.
4. Launch the application based on the deployments specified in **`microservices/hello_world/deployments.yaml`** and the gateway defined in **`microservices/hello_world/gateway.yaml`**.

You can modify the **`bolded`** files above for your settings to create a cluster for your application.


<!-- Setup VM -->

<!-- Graphic of the setup (from istio gke telemetry) -->

<!-- Creating Sample Application -->

<!-- Verifying things worked -->


## 2. COLA Training

The following script trains COLA for 500-2000 user (250-1000 requests per second) workload with a 50ms median latency target on the `Hello World` application you just launched.

    python3 examples/hello_world/hello_world_train.py

The script does the following:
1. Scales the number of nodes up in the application node pool.
2. Relaunches the application in case it is not running.
3. Trains COLA's model (for more details see the paper) for the request per second values as shown in **`configs/app/hello_world/train_config.py`**. The latency target is also configured in this file.
5. Maps each of the workloads to a "context" or request rate as logged in Google Cloud Monitoring.
4. Persists the models and the contexts to a location given in `configs/app/hello_world/train_config.py`.

You can modify the **`bolded`** files above for your settings to create a cluster for your application.

<!-- Settings for training (latency target, workloads) -->

<!-- Running training -->

<!-- Tracking progress -->


## 3. COLA Evaluation

The following script evaluates COLA on a set of workloads for the `Hello World` application that we trained.

    python3 examples/hello_world/hello_world_evaluate.py

The script does the following:
1. Runs the workloads defined in **`configs/app/hello_world/evaluation_config.py`**.
2. Records the results (number of VM instances used, latencies, failed requests) to **`evaluation/results/hello_world/<name>`** where <name> is defined in **`configs/app/hello_world/evaluation_config.py`**.

<!-- Settings for evaluation (workloads) -->

<!-- Running evaluation for comparables -->

<!-- Running evaluation for comparables -->

<!-- Viewing results for evaluation -->


## COLA Inference 

Once trained, COLA may be run in inference mode on a live cluster. Your application is serving live traffic to users and COLA will autoscale based on the current workload. We've included a couple of examples of running COLA in inference mode for sample applications.

<!-- Running COLA in inference mode -->

**Hello World:**

    python3 examples/hello_world/hello_world_inference.py

**Book Info:**

    python3 examples/book_info/book_info_inference.py

**Online Boutique:**

    python3 examples/online_boutique/online_boutique_inference.py

<!-- Tracking policy changes and workloads -->

