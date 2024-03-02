import os
import json
import time
import copy
import pickle
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import embed
import itertools

import scipy
import cvxpy as cvx
from scipy.stats import norm
import sklearn.gaussian_process as gp
from scipy.optimize import nnls, minimize
from scipy.optimize import minimize

from utils import kube_utils, cluster_utils, hpa_utils
from load_generator.locust_loadgen import LoadGenerator
from microservices.launch_apps import launch_application
from inference.cloud_metrics import CloudMetrics


logging.basicConfig(filename='logs/training.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('training')

# Class to Train Bandit Autoscaling algorithm.
class CherryPickTrainer(object):
    def __init__(self, name, config, train_config, search_history_len=3, hatch_rate=100, num_samples=5, sample_interval=1):

        # Initialize the configs for the deployment and training procedure.
        self.config = config
        self.train_config = train_config
        self.train_config.app_config = config
        
        # Set up action space.
        self.num_actions = {}
        for service, max_replicas in config.deployments.items():
            self.num_actions[service] = max_replicas        
                
        # Create a place to store configs.
        self.policy = {}
        for service in config.services:
            self.policy[service] = 1    
        kube_utils.apply_policy(self.policy)
        logger.info('Applied policy {}'.format(self.policy))
        
        # Model name and persistance details.
        self.name = name
        self.save_path = os.path.join('models', self.train_config.app_config.name, self.name)
        self.create_save_dirs()

        # Setup load generator for sampling qoe.
        self.hatch_rate = hatch_rate
        self.num_samples = num_samples
        self.sample_interval = sample_interval

        self.lg = LoadGenerator(host=self.config.host, 
                                locustfile=self.train_config.locustfile, 
                                duration=self.train_config.sample_duration,
                                csv_path=self.train_config.csv_path,
                                latency_threshold=self.train_config.latency_threshold,
                                w_i=self.train_config.w_i,
                                w_l=self.train_config.w_l,
                                hatch_rate=self.hatch_rate,
                                lat_opt=self.train_config.lat_opt,
                                )

        # Setup the node pool for training
        cluster_utils.scale_node_pool(
                                        cluster=self.train_config.cluster_name,
                                        project=self.train_config.project_name,
                                        zone=self.train_config.zone,
                                        node_pool=self.train_config.node_pool,
                                        num_nodes=self.train_config.max_nodes
                                    )
        logger.info('Scaled node pool {} to {} nodes'.format(self.train_config.node_pool, 
                                                              self.train_config.max_nodes))

        # Launch the application.
        launch_application(config=self.config)
        time.sleep(90)

    def run(self):

        # Get training data points.
        self.construct_search_space()

        # Evaluate training data points.
        #for context in self.train_config.train_rps:
        self.run_training()

        # Save model.
        self.save_model()

        # Tear down cluster.
        self.cleanup()


    def construct_search_space(self):

        self.bounds = []
        self.bounds.append([self.train_config.train_rps[0], self.train_config.train_rps[-1]])

        mcs_ranges = {}
        for deployment in self.config.deployments:
            mcs_ranges[deployment] = np.arange(1, self.config.deployments[deployment])
            self.bounds.append([1, self.config.deployments[deployment]])

        # Compose entire search space.
        self.search_coords = list(mcs_ranges.keys())
        #search_space = [self.train_config.train_rps] + list([x.tolist() for x in mcs_ranges.values()])
        #self.training_points = list(itertools.product(*search_space))
        #self.training_points = [list(x) for x in self.training_points]

        self.bounds = np.array(self.bounds)

        return


    def run_training(self, n_init=1):

        # Create initial samples to evaluate GP fn on.
        x_list = []
        y_list = []

        for _ in range(n_init):
            sample = sample_round_multivariate(self.bounds)
            x_list.append(sample)
            y_list.append(self.run_sample(sample))

        xp = np.array(x_list)
        yp = np.array(y_list)

        #embed()
        # Create Gaussian Process Regression model.
        kernel = gp.kernels.Matern()
        self.model = gp.GaussianProcessRegressor(kernel=kernel)

        # Iteratively estimate function which optimizes our loss.
        for n in range(self.num_samples):
            print("running sample {}".format(n))
            # Fit model on current data.
            self.model.fit(xp, yp)

            # Resample for next point
            proceed = False
            num_tries = 0
            while proceed is False and num_tries < 10:

                start = time.time()
                next_sample = sample_next_hyperparameter(expected_improvement, 
                self.model, yp, greater_is_better=True, bounds=self.bounds, n_restarts=100)
                end = time.time()
                print("Took {} seconds to get sample".format(end-start))

                # Check to see if we obtained a duplicate sample.
                #if np.any(np.abs(next_sample - xp) <= 1e-3):
                #    proceed = False
                #else:
                #    proceed = True
                proceed = True
                num_tries += 1

            next_sample = [int(x) for x in next_sample]

            x_list.append(next_sample)
            y_list.append(self.run_sample(next_sample))
            xp = np.array(x_list)
            yp = np.array(y_list)

        self.xp = xp
        self.yp = yp

        return


    def run_sample(self, sample):

        # Convert sample to deployment config.
        deployment_config = {}
        for i, num_replicas in enumerate(sample[1:]):
            deployment_config[self.search_coords[i]] = num_replicas

        kube_utils.apply_policy(deployment_config)
        time.sleep(45)

        # Evaluate action.
        self.lg.generate_load(rps=sample[0])
        reward, stats = self.lg.eval_qoe(rps=sample[0], action=sum(deployment_config.values()))

        return reward

    def predict(self, sample):
        return self.model.predict(np.array(sample).reshape(1,-1))


    def save_model(self):
        model_path = os.path.join(self.policy_path, 'policy.pk')
        pickle.dump(self, open(model_path, "wb"))
        return


    def cleanup(self):
        # Scale down training node pool.
        cluster_utils.scale_node_pool(
                                        cluster=self.train_config.cluster_name,
                                        project=self.train_config.project_name,
                                        zone=self.train_config.zone,
                                        node_pool=self.train_config.node_pool,
                                        num_nodes=0
                                    )
        return

    def create_save_dirs(self):

        # Make directory for the top level training routine.
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)

        # Make directory to save policy updates.
        self.policy_path = os.path.join(self.save_path, 'policy')
        if not os.path.exists(self.policy_path): 
            os.mkdir(self.policy_path)

        # Make directory to save bandit updates
        self.bandit_path = os.path.join(self.save_path, 'bandit')
        if not os.path.exists(self.bandit_path): 
            os.mkdir(self.bandit_path)

        # Create subfolder for each rps value we train on.
        for rps in self.train_config.train_rps:
            if not os.path.exists(os.path.join(self.policy_path, str(rps))):
                os.mkdir(os.path.join(self.policy_path, str(rps)))
            if not os.path.exists(os.path.join(self.bandit_path, str(rps))):
                os.mkdir(os.path.join(self.bandit_path, str(rps)))
        return





def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def sample_round_multivariate(bounds):

    sample = []
    for bound in bounds:
        sample.append(np.random.randint(bound[0],bound[1],1)[0])

    return sample
