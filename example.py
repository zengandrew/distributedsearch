from distributedsearch import DistributedSearch
from distributedsearch.util import NoLogging

from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
import numpy as np
import torch

def objective_function(rank, working_dir, params):
    images = Trajectory('train.traj')

    elements = np.unique([atom.symbol for atom in images[0]])
    cutoff = 6.0
    cosine_cutoff_params = {'cutoff_func': 'cosine'}
    gds = GaussianDescriptorSet(elements, cutoff, cosine_cutoff_params)

    g2_etas = [0.25, 2.5, 0.25, 2.5]
    g2_rs_s = [0.0, 0.0, 3.0, 3.0]
    gds.batch_add_descriptors(2, g2_etas, g2_rs_s, [])

    g4_etas = [0.005, 0.005, 0.01, 0.01]
    g4_zetas = [1.0, 4.0, 4.0, 16.0]
    g4_gammas = [1.0, 1.0, -1.0, -1.0]
    gds.batch_add_descriptors(4, g4_etas, g4_zetas, g4_gammas)

    amptorch_config = {
        'model': {
            'get_forces': True,
            'num_layers': params['num_layers'],
            'num_nodes': params['num_nodes'],
            'batchnorm': False,
        },
        'optim': {
            'force_coefficient': 0.04,
            'lr': 1e-2,
            'batch_size': 32,
            'epochs': 100,
            'loss': 'mse',
            'metric': 'mae',
            'gpus': 0,
        },
        'dataset': {
            'raw_data': images,
            'val_split': 0.1,
            'fp_params': gds,
            'save_fps': False,
            'scaling': {'type': 'normalize', 'range': (0, 1)},
        },
        'cmd': {
            'debug': False,
            'run_dir': working_dir,
            'seed': 1,
            'identifier': 'rank{}'.format(rank),
            'verbose': False,
            'logger': False,
        },
    }

    result = {}

    with NoLogging():
        torch.set_num_threads(1)
        trainer = AtomsTrainer(amptorch_config)
        trainer.train()

        predictions = trainer.predict(images)

        true_energies = np.array([image.get_potential_energy() for image in images])
        pred_energies = np.array(predictions['energy'])

        mse = np.mean((true_energies - pred_energies) ** 2)

        result = {
            'config': config,
            'mse': mse,
            'elapsed_time': (time.time() - start)
        }

    return result['mse']

def main():
    search_space = {
        'num_layers': range(3, 10),
        'num_nodes': range(5, 30)
    }

    search = DistributedSearch(search_space, objective_function)
    best_params, best_mse = search.get_minimum(trials=50)

    print('Best params', best_params)
    print('Best MSE', best_mse)

if __name__ == "__main__":
    main()
