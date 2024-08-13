import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

from config import num_qubits, range_gates

num_epochs = 50
batch_size = 512

f = open("logs.txt", "w")
for range_gates in tqdm(range_gates):
    directory = f"data/{num_qubits}_{range_gates}"
    file = np.loadtxt(f"{directory}/circuit.txt", delimiter=",")
    circuits = np.zeros(shape=(len(file), 6, num_qubits, range_gates // num_qubits))
    for circuit in range(len(file)):
        for qubit in range(num_qubits):
            for gate in range(range_gates // num_qubits):
                circuits[circuit][int(file[circuit][gate])][qubit][gate] = 1

    circuits = torch.tensor(circuits, dtype=torch.float32)

    file = np.loadtxt(f"{directory}/real.txt", delimiter=",")
    reals = torch.tensor(file, dtype=torch.float32)

    file = np.loadtxt(f"{directory}/imaginary.txt", delimiter=",")
    imaginaries = torch.tensor(file, dtype=torch.float32)

    file = np.loadtxt(f"{directory}/probability.txt", delimiter=",")
    probabilities = torch.tensor(file, dtype=torch.float32)

    device = "cuda"

    kernel_size = 2
    real_model = nn.Sequential(
        nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=kernel_size, stride=1, padding=1
        ),
        nn.BatchNorm2d(32),
        nn.Mish(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(32),
        nn.Mish(),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(64),
        nn.Mish(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(64),
        nn.Mish(),
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(in_features=256, out_features=300),
        nn.BatchNorm1d(300),
        nn.Mish(),
        nn.Linear(in_features=300, out_features=100),
        nn.BatchNorm1d(100),
        nn.Mish(),
        nn.Linear(in_features=100, out_features=50),
        nn.BatchNorm1d(50),
        nn.Mish(),
        nn.Linear(in_features=50, out_features=2**num_qubits),
        nn.Tanh(),
    ).cuda()

    imaginary_model = nn.Sequential(
        nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=kernel_size, stride=1, padding=1
        ),
        nn.BatchNorm2d(32),
        nn.Mish(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(32),
        nn.Mish(),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(64),
        nn.Mish(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(64),
        nn.Mish(),
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(128),
        nn.Mish(),
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(256),
        nn.Mish(),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(in_features=256, out_features=300),
        nn.BatchNorm1d(300),
        nn.Mish(),
        nn.Linear(in_features=300, out_features=100),
        nn.BatchNorm1d(100),
        nn.Mish(),
        nn.Linear(in_features=100, out_features=50),
        nn.BatchNorm1d(50),
        nn.Mish(),
        nn.Linear(in_features=50, out_features=2**num_qubits),
        nn.Tanh(),
    ).cuda()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(real_model.parameters(), lr=0.001)
    optimizer_2 = optim.Adam(imaginary_model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        for i in range(0, len(circuits), batch_size):
            Xbatch = circuits[i : i + batch_size]
            y_pred = real_model(Xbatch.cuda())
            y_pred_2 = imaginary_model(Xbatch.cuda())
            ybatch = reals[i : i + batch_size]
            ybatch_2 = imaginaries[i : i + batch_size]
            loss = loss_fn(y_pred.cuda(), ybatch.cuda())
            loss_2 = loss_fn(y_pred_2.cuda(), ybatch_2.cuda())
            optimizer.zero_grad()
            optimizer_2.zero_grad()
            loss.backward()
            loss_2.backward()
            optimizer.step()
            optimizer_2.step()

    sample_size = 1024
    circuit = random.randint(0, len(circuits) - batch_size)
    real_angles = torch.atan2(
        reals[circuit : circuit + batch_size].cuda(),
        real_model(circuits[circuit : circuit + batch_size].cuda()),
    )
    imaginary_angles = torch.atan2(
        imaginaries[circuit : circuit + batch_size].cuda(),
        imaginary_model(circuits[circuit : circuit + batch_size].cuda()),
    )

    angles = torch.abs(real_angles - imaginary_angles)
    angles = torch.min(angles, 2 * torch.pi - angles)

    f.write(f"{angles.cpu().detach().numpy().tolist()}")
    f.write(
        f"{torch.mean(angles).cpu().detach().numpy()}, {torch.std(angles).cpu().detach().numpy()}\n"
    )

    batch_size = 12
    print(
        f"Actual Reals: {reals[circuit : circuit + batch_size]}, Actual Imaginaries: {imaginaries[circuit : circuit + batch_size]}"
    )
    print(
        f"Simulated Reals: {real_model(circuits[circuit : circuit + batch_size].cuda())}), Simulated Imaginaries: {imaginary_model(circuits[circuit : circuit + batch_size].cuda())})"
    )
