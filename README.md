# distributed-search
Distributed hyperparameter search

## Setup
```
git clone https://github.com/ulissigroup/amptorch.git
cd amptorch
conda env create -f env_cpu.yml
conda activate amptorch
pip install -e .
conda install mpi4py
```

## Usage (PBS)
```
git clone https://github.com/zengandrew/distributed-search.git
qsub job.pbs
```
