"""
Causal Quantum Feature Extractor (CausalQuantumPolicy) for Autonomous IoT Systems.
Combines Structural Causal Latent Representations with PennyLane Variational Quantum Circuits (VQCs).
"""

import torch
import torch.nn as nn
import pennylane as qml
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CausalQuantumPolicy(BaseFeaturesExtractor):
    """
    Hybrid Causal-Quantum Policy Feature Extractor.
    
    Architecture:
        1. Causal Latent Encoder: Linear(54 -> 128) -> ReLU -> Linear(128 -> N_qubits) -> Tanh.
        2. Variational Quantum Circuit (VQC):
           - N_qubits (default 4 qubits, 5 layers).
           - Data Re-uploading scheme with Ry AngleEmbedding.
           - StronglyEntanglingLayers / EfficientSU2 ansatz with circular CNOT entangling gates.
           - Measurement: Pauli-Z expectation values across all qubits.
        3. Latent Head: Linear(N_qubits -> 64) -> ReLU -> Linear(64 -> Features_dim) -> ReLU.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 8, n_qubits: int = 4, n_layers: int = 5):
        super().__init__(observation_space, features_dim)

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Learnable input scaling
        self.input_scaling = nn.Parameter(torch.ones(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # 1. Causal Latent Feature Encoder
        self.causal_encoder = nn.Sequential(
            nn.Linear(54, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
            nn.Tanh()
        )

        # 2. Parameterized Quantum Circuit QNode
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights, scaling):
            for l in range(n_layers):
                scaled_inputs = inputs * scaling
                qml.AngleEmbedding(scaled_inputs, wires=range(n_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights[l], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {
            "weights": (n_layers, 1, n_qubits, 3),
            "scaling": (n_qubits,)
        }

        self.vqc = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 3. Post-processing Latent Head
        self.post_processing = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping IoT state to causal quantum latent representation."""
        z_causal = self.causal_encoder(observations)
        q_expval = self.vqc(z_causal)
        return self.post_processing(q_expval)
