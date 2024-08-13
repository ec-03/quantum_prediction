from typing import Tuple, List
import cirq
from gate import Gate, GateType
import numpy as np


class Circuit:
    def __init__(self, qubit_count) -> None:
        self.cirq_circuit = cirq.Circuit()
        self.gate_types = []
        self.qubits = cirq.LineQubit.range(qubit_count)

    def add_gate(self, gate: Gate) -> None:
        match gate.gate_type.qubits_operated:
            case 2:
                self.gate_types.append(gate.gate_type.value)
                self.gate_types.append(gate.gate_type.value)
            case 1:
                self.gate_types.append(gate.gate_type.value)
        self.cirq_circuit.append(gate.operate())

    def __str__(self) -> str:
        # print them by moment, "snaking" down
        # return ",".join(str(gate_type) for gate_type in self.gate_types)

        # print them by qubit
        output = []
        for i in range(len(self.qubits)):
            for j in range(len(self.gate_types) // len(self.qubits)):
                output.append(str(self.gate_types[i + j * len(self.qubits)]))
        return ",".join(output)

    def simulate(self, simulator) -> Tuple[List[float]]:
        state_vector = simulator.simulate(self.cirq_circuit).state_vector()
        return (
            [state.real for state in state_vector],
            [state.imag for state in state_vector],
        )
