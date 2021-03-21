from .util import RestoreWorkingDir

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import os
import tempfile
import time

class ObjectiveFunctionWrapper:
    def __init__(self, objective_function, logging=False):
        self.raw_objective_function = objective_function
        self.logging = logging

    def __call__(self, params):
        objective = None
        duration = None

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        with tempfile.TemporaryDirectory(prefix='rank{}_'.format(rank)) as scratch_dir:
            start = time.time()

            with RestoreWorkingDir():
                objective = self.raw_objective_function(rank, scratch_dir, params)

            duration = time.time() - start

        info = {
            'params': params,
            'duration': duration
        }

        if self.logging:
            print('[{}] objective: {} | params: {} | duration: {}'.format(rank, objective, params, duration))

        return objective, info

class DistributedSearch:
    def __init__(self, search_space, objective_function, logging=False):
        self.search_space = search_space
        self.objective_function = ObjectiveFunctionWrapper(objective_function, logging)

    def _generate_trial_configurations(self, size):
        configs = []

        for i in range(size):
            config = {}
            for hyperparameter in self.search_space:
                if isinstance(self.search_space[hyperparameter], range):
                    config[hyperparameter] = int(np.random.choice(self.search_space[hyperparameter]))
            configs.append(config)

        return configs

    def get_minimum(self, trials=25):
        trial_configurations = self._generate_trial_configurations(trials)
        
        minimum_params = None
        minimum_objective = None

        # unused
        results = []

        with MPIPoolExecutor() as executor:
            results_generator = executor.map(self.objective_function, trial_configurations, unordered=True)

            for result in results_generator:
                objective, info = result

                if minimum_objective is None or objective < minimum_objective:
                    minimum_params = info['params']
                    minimum_objective = objective

                results.append(result)

        return minimum_params, minimum_objective