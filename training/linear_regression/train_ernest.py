import os
import sys
sys.path.insert(1, os.getcwd())

import time
import pickle
import random
import itertools
import logging
import numpy as np
import cvxpy as cvx
from scipy.optimize import nnls
import sklearn.linear_model.LinearRegression as LinearRegression

import utils.kube as kube_utils
import utils.cluster as cluster_utils
from utils.launch_apps import launch_application
from utils.locust_loadgen import LoadGenerator

logging.basicConfig(filename='logs/training.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('training')

# Class to Train Bandit Autoscaling algorithm.
class ErnestTrainer(object):
    def __init__(self, name, config, train_config, search_history_len=3, hatch_rate=100, num_samples=3, sample_interval=1, sample_strategy='opt_design'):

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

        # Services we searched recently.
        self.search_history = []
        self.search_history_len = search_history_len # Number of recently searched services to exclude for next search.
        
        # Number of training points.
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.sample_strategy = sample_strategy

        # Model name and persistance details.
        self.name = name
        self.save_path = os.path.join('models', self.train_config.app_config.name, self.name)
        self.create_save_dirs()

        # Setup load generator for sampling qoe.
        self.hatch_rate = hatch_rate
        #embed()

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

        print('Scaled node pool {} to {} nodes'.format(self.train_config.node_pool,
                                                              self.train_config.max_nodes))
        # Launch the application.
        launch_application(config=self.config)
        time.sleep(90)


    def run(self):

        # Get training data points.
        print("Started expt design")
        self.expt_design()

        # Evaluate training data points.
        print("Started running training")
        self.run_training(num_samples=self.num_samples)

        # Build predictive model.
        self.build_predictor()

        # Save model.
        self.save_model()

        # Tear down cluster.
        self.cleanup()


    def _construct_constraints(self, lambdas, points, budget=10):
        '''Construct non-negative lambdas and budget constraints'''
        constraints = []
        constraints.append(0 <= lambdas)
        constraints.append(lambdas <= 1)
        # constraints.append(self._get_cost(lambdas, points) <= budget)
        return constraints


    def expt_design(self):
        """ Produce data points to run w/ optimal experiment design"""

        training_points = self.get_training_points()
        num_points = len(training_points)


        if self.sample_strategy == 'random':
            print("Random sample")
            random.shuffle(training_points)
            self.sorted_training_points = [(x, i) for (i,x) in enumerate(training_points)]
            return

        all_training_features = np.array([_get_features(point) for point in training_points])
        covariance_matrices = list(_get_covariance_matrices(all_training_features))

        #covariance_matrices = np.array(covariance_matrices).astype(np.double)
        lambdas = cvx.Variable(num_points)
        objective = cvx.Minimize(_construct_objective(covariance_matrices, lambdas))
        constraints = self._construct_constraints(lambdas, training_points)

        problem = cvx.Problem(objective, constraints)

        start = time.time()
        opt_val = problem.solve(solver='CVXOPT')
        end = time.time()
        print("took {} seconds".format(end-start))
        # TODO: Add debug logging
        # print "solution status ", problem.status
        # print "opt value is ", opt_val

        filtered_lambda_idxs = []
        for i in range(0, num_points):
            if lambdas[i].value > .3:
                filtered_lambda_idxs.append((lambdas[i].value, i))

        sorted_by_lambda = sorted(filtered_lambda_idxs, key=lambda t: t[0], reverse=True)
        self.sorted_training_points = [(training_points[idx], l) for (l, idx) in sorted_by_lambda]
        return

    def get_training_points(self):

        # Get machine range for each service.
        mcs_ranges = {}
        for deployment in self.config.deployments:
            mcs_ranges[deployment] = np.arange(1, int(self.config.deployments[deployment] / float(self.sample_interval)) )

        # Compose entire search space.
        self.search_coords = list(mcs_ranges.keys())
        search_space = [self.train_config.train_rps] + list([x.tolist() for x in mcs_ranges.values()])
        self.training_points = list(itertools.product(*search_space))
        self.training_points = [list(x) for x in self.training_points]
        print("Total training points: {}".format(len(self.training_points)))

        return self.training_points


    def run_training(self, num_samples = 3):

        # Iterate through training points and record performance.
        self.res = []
        for (training_point, _lambda) in self.sorted_training_points[:num_samples]:
            context = training_point[0]
            print("Training point: {}".format(training_point))

            # Get deployment config.
            deployment_config = {}
            for i,num_replicas in enumerate(training_point[1:]):
                deployment_config[self.search_coords[i]] = num_replicas*self.sample_interval
                
            # Scale deployment config.
            kube_utils.apply_policy(deployment_config)
            time.sleep(45)

            # Evaluate action.
            self.lg.generate_load(rps=context)
            reward, stats = self.lg.eval_qoe(rps=context, action=sum(deployment_config.values()))

            # Record results.
            self.res.append({'training_point': training_point, 'reward': reward, 'stats': stats})

        return


    def predict(self, training_point):
        test_features = np.array(_get_features(training_point))
        return test_features.dot(self.model[0])
    
    def build_predictor(self):

        # Process results from training for model fitting.
        print("Fitting a model with ", len(self.res), " points")
        labels, data_points = [], []
        for training_example in self.res:
            labels.append(training_example['stats'][self.train_config.lat_opt]) # Latency
            data_points.append(_get_features(training_example['training_point']))

        # Fit model.
        self.model = nnls(data_points, labels)

        # Calculate training error
        #training_errors = []
        #for p in self.training_data:
        #    predicted = self.predict(p[0], p[1])
        #    training_errors.append(predicted / p[2])

        #training_errors = [str(np.around(i*100, 2)) + "%" for i in training_errors]
        #print("Prediction ratios are", ", ".join(training_errors))
        return self.model[0]


    def build_ols_predictor(self):
        # Process results from training for model fitting.
        print("Fitting a model with ", len(self.res), " points")
        labels, data_points = [], []
        for training_example in self.res:
            labels.append(training_example['stats'][self.train_config.lat_opt]) # Latency
            data_points.append(_get_features(training_example['training_point']))

        # Fit Ordinary Least Squares model.
        self.ols_model = LinearRegression()
        self.ols_model.fit(data_points, labels)

        return

    def predict_ols(self, training_point):
        reward = self.ols_model.predict([_get_features(training_point)])
        return reward.tolist()[0]

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


def _construct_objective(covariance_matrices, lambdas):
    ''' Constructs the CVX objective function. '''
    num_points = len(covariance_matrices)
    num_dim = int(covariance_matrices[0].shape[0])
    objective = 0.0
    matrix_part = np.zeros([num_dim, num_dim])
    for j in np.arange(0, num_points):
        matrix_part = matrix_part + covariance_matrices[j] * lambdas[j]

    for i in np.arange(0, num_dim):
        k_vec = np.zeros(num_dim)
        k_vec[i] = 1.0
        objective = objective + cvx.matrix_frac(k_vec, matrix_part)

    return objective


def _get_covariance_matrices(features_arr):
    ''' Returns a list of covariance matrices given expt design features'''
    col_means = np.mean(features_arr, axis=0) + 1e-3
    means_inv = (1.0 / col_means)
    nrows = features_arr.shape[0]
    for i in np.arange(0, nrows):
        feature_row = features_arr[i,]
        ftf = np.outer(feature_row.transpose(), feature_row)
        yield np.diag(means_inv).transpose().dot(ftf.dot(np.diag(means_inv)))

def _get_features(training_point):
    ''' Compute the features for a given point. Point is expected to be [input_frac, machines]'''
    features = [1.0]
    scale = training_point[0]
    for mcs in training_point[1:]:
        features.append(float(scale) / float(mcs))
        features.append(float(mcs))
        features.append(np.log(mcs))
    return features
