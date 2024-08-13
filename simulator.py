from typing import List, Tuple
import qsimcirq
from generator import generate
from gate import GateType
import os
from tqdm import tqdm
import shutil


def simulate(
    num_qubits: int,
    num_gates: int,
    num_circuits: int,
    gate_set: List[Tuple[GateType, int]],
):
    simulator = qsimcirq.QSimSimulator()

    directory = f"./data/{num_qubits}_{num_gates}"

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    circuit_file = open(f"{directory}/circuit.txt", "w")
    real_file = open(f"{directory}/real.txt", "w")
    imaginary_file = open(f"{directory}/imaginary.txt", "w")
    probability_file = open(f"{directory}/probability.txt", "w")

    for i in tqdm(range(num_circuits)):
        circuit = generate(num_qubits, num_gates, gate_set)
        (real, imaginary) = circuit.simulate(simulator)

        circuit_file.write(str(circuit) + "\n")
        real_file.write(",".join(["%.7f" % number for number in real]) + "\n")
        imaginary_file.write(",".join(["%.7f" % number for number in imaginary]) + "\n")
        probability_file.write(
            ",".join(
                [
                    "%.7f"
                    % min(
                        1.0, real[index] ** 2 + imaginary[index] ** 2
                    )  # occasionally rounding makes this slightly above 1
                    for index in range(len(real))
                ]
            )
            + "\n"
        )
