# quantum_prediction

## Setup
Install cirq, tqdm, torch, numpy, qsimcirq. Setup cuda support.
NOTE: I can not get qsimcirq to compile on a ARM Mac.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install numpy cirq tqdm torch pybind11 qsimcirq
```

## Building dataset
Change the parameters set in `config.py`.
gate_set is a list of tuples. Each tuple is one gate type as defined in `gate.py`, and one weight for its occurrance in the circuit.
Then, run
```shell
python3 build_dataset.py
```
Data will be generated in the `data/` directory.

## Training and evaluating the model
Right now, only very basic methods of evaluating the model are implemented.
Change the parameters (`batch_size`, `num_epochs`) in train.py. Optionally change the models as well.
```shell
python3 train.py
```