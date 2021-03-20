from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from mpi4py import MPI
import numpy as np
import os
import sys
import tempfile
import time
import torch

# https://stackoverflow.com/a/45669280
class NoLogging:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w') # standard printing
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w') # tqdm printing

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout
        sys.stderr.close()
        sys.stderr = self.stderr

def train_task(train_traj, config):
    start = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    with tempfile.TemporaryDirectory(prefix='rank{}_'.format(rank)) as temp_dir:
        print('[rank{}@{}] received {}'.format(rank, temp_dir, config))

        images = Trajectory(train_traj)

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
                'num_layers': config['num_layers'],
                'num_nodes': config['num_nodes'],
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
                'run_dir': temp_dir,
                'seed': 1,
                'identifier': 'rank{}'.format(rank),
                'verbose': False,
                'logger': False,
            },
        }

        torch.set_num_threads(1)

        predictions = None

        with NoLogging():
            trainer = AtomsTrainer(amptorch_config)
            trainer.train()
            predictions = trainer.predict(images)

        if predictions is not None:
            true_energies = np.array([image.get_potential_energy() for image in images])
            pred_energies = np.array(predictions['energy'])

            mse = np.mean((true_energies - pred_energies) ** 2)

            result = {
                'config': config,
                'mse': mse,
                'elapsed_time': (time.time() - start)
            }

    return result
