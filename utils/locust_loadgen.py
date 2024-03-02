import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import utils.kube as kube_utils

class LoadGenerator(object):
    def __init__(self, host, locustfile, duration=25, csv_path='logs/scratch/cola_lg', 
                num_workers=10, hatch_rate=100, latency_threshold=100, w_i=5, w_l=10,
                lat_opt='Average Latency'):

        # Setup config for load generation.
        self.host = host
        self.locustfile = locustfile
        self.duration = duration
        self.csv_path = csv_path
        self.num_workers = num_workers
        self.hatch_rate = hatch_rate

        # QOE parameters.
        self.latency_threshold = latency_threshold
        self.w_i = w_i
        self.w_l = w_l
        self.lat_opt = lat_opt

        # Increase file limit and kill any currently running load generation.
        increase_open_files()
        kill_load_gen()

    def generate_load(self, rps, duration=None):
        ''' Generate load for a given rps '''

        if duration is None:
            duration = self.duration

        # Run load generation.
        cmd = 'locust -f {locustfile} --csv-full-history --csv={csv_path}.csv --headless -u {rps} -r {hatch_rate} --host {host} --run-time {duration}s'.format(
            csv_path=self.csv_path, rps=rps, host=self.host, duration=duration, hatch_rate=self.hatch_rate, locustfile=self.locustfile)
        print(cmd)
        os.system(cmd)
        kill_load_gen()

        return


    def generate_load_distributed(self, rps, duration=None):
        ''' Generate load for a given rps '''

        if duration is None:
            duration = self.duration

        # Start Master for Locust.
        cmd = 'locust -f {locustfile} --master --csv-full-history --only-summary --csv={csv_path}.csv --headless -u {rps} -r {hatch_rate} --expect-workers {num_workers} --host {host} --run-time {duration}s --logfile /tmp/locust.log --loglevel ERROR >>locustlog.txt 2>&1 &'.format(
            csv_path=self.csv_path, rps=rps, host=self.host, duration=duration, hatch_rate=self.hatch_rate, locustfile=self.locustfile, num_workers=self.num_workers)
        os.system(cmd)

        # Start Workers.
        for worker in range(self.num_workers):
            os.system(
                'locust -f {locustfile} --worker --host {host} --logfile /tmp/locust_worker.log --loglevel ERROR >>locustlog.txt 2>&1 &'.format(host=self.host, locustfile=self.locustfile))

        # Wait for execution to finish and kill load generator.
        time.sleep(self.duration+15)
        kill_load_gen()

        return
    

    def read_load_statistics(self, rps):

        # Read output from locust.
        df = pd.read_csv('{}.csv_stats.csv'.format(self.csv_path))

        # Create result dictionary.
        stats = {}
        stats['Average Latency'] = df['50%'].iloc[-1]
        stats['90p Latency'] = df['90%'].iloc[-1]
        stats['95p Latency'] = df['95%'].iloc[-1]
        stats['99p Latency'] = df['99%'].iloc[-1]
        stats['Output RPS'] = df['Requests/s'].iloc[-1]
        stats['Failures RPS'] = df['Failures/s'].iloc[-1]
        stats['Input RPS'] = rps

        return stats

    def read_last_load_statistics(self, rps, num_rows=3):

        # Read last second output from locust.
        df = pd.read_csv('{}.csv_stats_history.csv'.format(self.csv_path))
        df = df[df['Name'] == 'Aggregated'].iloc[-num_rows:]
        df = df.mean()

        # Create result dictionary.
        stats = {}
        stats['Average Latency'] = df['50%']
        stats['90p Latency'] = df['90%']
        stats['95p Latency'] = df['95%']
        stats['99p Latency'] = df['99%']
        stats['Output RPS'] = df['Requests/s']
        stats['Failures RPS'] = df['Failures/s']
        stats['Input RPS'] = rps

        return stats


    def eval_qoe(self, rps, action):

        stats = self.read_load_statistics(rps)
        qoe = self.compute_qoe(stats, action)

        return qoe, stats


    def eval_last_load_qoe(self, rps, action):
    
        stats = self.read_last_load_statistics(rps)
        qoe = self.compute_qoe(stats, action)

        return qoe, stats


    def compute_qoe(self, stats, action):
        '''
        Sample qoe function for evaluation.
        Maximize RPS Coverage and Minimize Latency Slowdown
        '''    
        soft_constraint_latency = (self.latency_threshold - stats[self.lat_opt])
        
        # Hinge loss for the latency constraint.
        if soft_constraint_latency > 0:
            soft_constraint_latency = 0

        return soft_constraint_latency*self.w_l - int(action)*self.w_i


    def run_workload(self, rps_rates):

        # Run a workload, defined as a sequence of rps rates.
        iter_results = []
        for i in range(len(rps_rates)):
            
            # Start measuring the replicas allocated.
            self.measure_replicas()

            # Get performance statistics.
            start_time = datetime.now()
            self.generate_load(rps_rates[i])
            stats = self.read_load_statistics(rps_rates[i])
            print(stats)
            end_time = datetime.now()

            # Get current deployments.
            try:
                deployments = kube_utils.get_current_deployments()
            except:
                deployments = {}
            # Write CPU Utilization Statistics.
            try:
                utilization = kube_utils.get_pod_statistics()
                utilization = utilization.to_dict('records')
            except:
                utilization = None
            # Collect replica time series and add to the results.
            try:
                replicas = self.get_replicas()
            except:
                replicas = []

            # Add results for iteration.
            iter_results.append({
                                'iter': i, 
                                'perf': stats,
                                'rps_rate': rps_rates[i], 
                                'services': deployments, 
                                'utilization': utilization,
                                'start_time': start_time,
                                'end_time': end_time,
                                'replicas': replicas,
                                })

        return iter_results
        

    def measure_replicas(self):
    
        # Kill any measurement scripts still running.
        os.system('pkill -f "measure_replicas"')

        # Monitor number of deployments.
        os.system('python3 evaluation/eval_utils/measure_replicas.py &')

        return


    def get_replicas(self):

        # Read replica data.
        with open('service_replicas_count.json', 'r') as f:
            lines = f.readlines()
            replicas = [json.loads(x) for x in lines]

        # Stop measuring replica counts.
        os.system('pkill -f "measure_replicas"')

        return replicas


def kill_load_gen():
    os.system('pkill -f locust')
    return


def increase_open_files():
    # Increase number of open files to allow locust to open a large connection pool.
    os.system('ulimit -Sn 65535')
