from gate import GateType

num_qubits = 3
range_gates = [6, 9, 12, 15, 18, 21, 24, 27, 30]
num_circuits = 10000
gate_set = [
    (GateType.S, 1 / 5),
    (GateType.H, 1 / 5),
    (GateType.CNOT1, 1 / 10),
    (GateType.CNOT2, 1 / 10),
    (GateType.T, 1 / 5),
    (GateType.I, 1 / 5),
]
