#####################################
# Hello World example for COLA.
#####################################
import os, sys
sys.path.insert(1, os.getcwd())

# Load necessary configs into memory.
from configs.app.book_info.config import cfg4
from configs.app.book_info.train_config import train_cfg4
from configs.app.book_info.evaluation_config import eval_cfg4


#########
# Run 1 minute evaluations
#########

# Create instance of COLA class.
from main.autoscale import Autoscaler
cola = Autoscaler(config=cfg4, train_config=train_cfg4, eval_config=eval_cfg4, auth=True)

cola.create_cluster()
cola.evaluate(method='cpu')
cola.delete_cluster()
