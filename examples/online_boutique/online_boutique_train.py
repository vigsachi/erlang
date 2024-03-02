#####################################
# Hello World example for COLA.
#####################################
import os, sys
sys.path.insert(1, os.getcwd())

# Load necessary configs into memory.
from configs.app.online_boutique.config import cfg
from configs.app.online_boutique.train_config import train_cfg
from configs.app.online_boutique.evaluation_config import eval_cfg


# Create instance of COLA class.
from main.autoscale import Autoscaler
cola = Autoscaler(config=cfg, train_config=train_cfg, eval_config=eval_cfg, auth=True)

# Launch GKE application.
cola.train(method='cola', run_name='train_test')