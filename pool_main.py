from worker import train_task

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np

# number of hyperparameter search space samples to evaluate
TRIALS = 50

# search space definition
SEARCH_SPACE = {
    'num_layers': range(3, 10),
    'num_nodes': range(5, 30)
}

def generate_trial_configurations():
    configs = []

    for i in range(TRIALS):
        config = {}
        for hyperparameter in SEARCH_SPACE:
            config[hyperparameter] = int(np.random.choice(SEARCH_SPACE[hyperparameter]))
        configs.append(config)

    return configs

def main():
    # sample from hyperparameter search space
    print('===== Configurations =====')
    configs = generate_trial_configurations()
    print(configs)
    print('==========')

    # train and evaluate models
    results = []
    with MPIPoolExecutor() as executor:
        results_generator = executor.map(train_task, ['train.traj'] * TRIALS, configs)

        for result in results_generator:
            results.append(result)

    # summarize results
    print('===== Results =====')
    for count, result in enumerate(results):
        print('Result {}: {}'.format(count, result))
    print('==========')

if __name__ == "__main__":
    main()
