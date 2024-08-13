import cirq
import numpy as np
from enum import Enum, auto


class GateType(Enum):
    I = 0
    S = 1
    H = 2
    T = 3
    CNOT1 = 4
    CNOT2 = 5

    @property
    def qubits_operated(self) -> int:
        match self:
            case GateType.CNOT1 | GateType.CNOT2:
                return 2
            case _:
                return 1


class Gate:
    def __init__(self, gate_type: GateType, qubits, angle: float = np.pi / 2) -> None:
        self.gate_type = gate_type
        self.qubits = qubits
        self.angle = angle

    def operate(self) -> cirq.GateOperation:
        match self.gate_type:
            case GateType.I:
                return cirq.I(self.qubits[0])
            case GateType.H:
                return cirq.H(self.qubits[0])
            case GateType.S:
                return cirq.S(self.qubits[0])
            case GateType.T:
                return cirq.T(self.qubits[0])
            case GateType.CNOT1:
                return cirq.CNOT(self.qubits[0], self.qubits[1])
            case GateType.CNOT2:
                return cirq.CNOT(self.qubits[1], self.qubits[0])

    def __str__(self) -> str:
        return f"Gate {self.gate_type.name} on qubits {self.qubits} with angle {self.angle}"
