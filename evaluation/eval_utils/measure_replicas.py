import os
import sys
import json
import time

sys.path.insert(0, os.getcwd())

from utils.kube import get_current_deployments, get_current_nodes

# File name, remove the old log on start
fname = 'service_replicas_count.json'
wait_time = 10 # Interval at which we get results


def run():
    
    # Remove log file if it exists.
    os.system('rm {}'.format(fname))

    while True:
        try:
            deployments = get_current_deployments()
            nodes = get_current_nodes()
            res_dict = {
                        'deployments': deployments,
                        'services': list(deployments.keys()), 
                        'tot_replicas': sum(deployments.values()), 
                        'nodes': nodes,
                        'time': time.time(),
                        }
            with open(fname, 'a+') as obj:
                obj.write(json.dumps(res_dict) + '\n')
            time.sleep(wait_time)

        except:
            pass


if __name__ == "__main__":
    run()
