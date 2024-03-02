#####################################
# Hello World example for COLA.
#####################################
import os, sys
sys.path.insert(1, os.getcwd())

# Load necessary configs into memory.
from configs.app.hello_world.config import cfg
from configs.app.hello_world.train_config import train_cfg
from configs.app.hello_world.evaluation_config import eval_cfg


# Create instance of COLA class.
from main.autoscale import Autoscaler
cola = Autoscaler(config=cfg, train_config=train_cfg, eval_config=eval_cfg, auth=True)

# Create a cluster on GKE.
cola.create_cluster()

# Authenticate to the GKE cluster.
cola.auth_cluster()

# Launch GKE application.
cola.launch_application()