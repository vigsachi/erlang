#####################################
# Hello World example for COLA.
#####################################
import os, sys
sys.path.insert(1, os.getcwd())

# Load necessary configs into memory.
from configs.app.book_info.config import cfg2
from configs.app.book_info.train_config import train_cfg2
from configs.app.book_info.evaluation_config import eval_cfg2


#########
# Run 1 minute evaluations
#########

# Create instance of COLA class.
from main.autoscale import Autoscaler
cola = Autoscaler(config=cfg2, train_config=train_cfg2, eval_config=eval_cfg2, auth=True)

cola.create_cluster()
cola.evaluate(method='cpu')
cola.delete_cluster()
