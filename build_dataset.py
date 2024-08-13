import qsimcirq
from gate import GateType
from simulator import simulate

import os

from config import num_qubits, num_circuits, gate_set, range_gates

if not os.path.exists("./data"):
    os.mkdir("./data")

for range_gates in range_gates:
    simulate(num_qubits, range_gates, num_circuits, gate_set)
