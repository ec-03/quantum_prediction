import random
from typing import List
import numpy as np
import cirq
from circuit import Circuit
from gate import Gate, GateType


def generate(
    qubit_count: int,
    gate_count: int,
    weighted_gate_type_set: list[tuple[GateType, float]],
) -> Circuit:
    circuit = Circuit(qubit_count)
    gate_types = np.random.choice(
        [gate[0] for gate in weighted_gate_type_set],
        5 * gate_count,
        [gate[1] for gate in weighted_gate_type_set],
    )  # weighted random sample with replacement
    index = qubit_count
    position = qubit_count

    for i in range(qubit_count):
        circuit.add_gate(Gate(GateType.H, [circuit.qubits[i]]))

    while position < gate_count:
        gate_type = gate_types[index]
        if (
            gate_type.qubits_operated == 2 and position % qubit_count == qubit_count - 1
        ):  # don't allow two-qubit gates to occur at the bottom of a column
            pass
        else:
            gate = Gate(
                gate_type,
                circuit.qubits[
                    position % qubit_count : position % qubit_count
                    + gate_type.qubits_operated
                ],
            )
            position += gate_type.qubits_operated
            circuit.add_gate(gate)
        index += 1
    return circuit
