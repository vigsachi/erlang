#####################################
# Hello World example for COLA.
#####################################
import os, sys
sys.path.insert(1, os.getcwd())

# Load necessary configs into memory.
from configs.app.book_info.config import cfg
from configs.app.book_info.train_config import train_cfg
from configs.app.book_info.evaluation_config import eval_cfg


#########
# Run 1 minute evaluations
#########

# Create instance of COLA class.
from main.autoscale import Autoscaler
cola = Autoscaler(config=cfg, train_config=train_cfg, eval_config=eval_cfg, auth=True)

# Authenticate to the GKE cluster.
cola.auth_cluster()

# Evaluate performance of the learned autoscaler.
#cola.evaluate(method='na')
# cola.evaluate(method='cpu')
cola.evaluate(method='cola')


