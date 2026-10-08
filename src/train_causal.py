"""
Training Script for Causal Quantum Deep Reinforcement Learning (CQDRL) Agents on Autonomous IoT Environments.
Compares: Causal-Quantum-DQN (CQDQN), Causal-Quantum-PPO (CQPPO), Standard Quantum-DQN, Classical-DQN.
"""

import os
import argparse
from stable_baselines3 import DQN, PPO

from iot_environment import AutonomousIoTSystemEnv
from cqdrl_policy import CausalQuantumPolicy


def train_causal_agents(total_timesteps=5000, data_path=None, output_dir="results"):
    """Trains and benchmarks causal quantum vs classical RL agents."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Causal IoT Environment
    env_causal = AutonomousIoTSystemEnv(data_path=data_path, use_causal_rewards=True)
    env_standard = AutonomousIoTSystemEnv(data_path=data_path, use_causal_rewards=False)

    policy_kwargs_quantum = dict(
        features_extractor_class=CausalQuantumPolicy,
        features_extractor_kwargs=dict(features_dim=8, n_qubits=4, n_layers=5),
    )

    models = {}

    print("=== 1. Training Causal-Quantum-DQN (CQDQN) ===")
    models["CQDQN"] = DQN("MlpPolicy", env_causal, policy_kwargs=policy_kwargs_quantum, learning_rate=1e-3, verbose=0)
    models["CQDQN"].learn(total_timesteps=total_timesteps)
    models["CQDQN"].save(os.path.join(output_dir, "CQDQN.zip"))

    print("=== 2. Training Causal-Quantum-PPO (CQPPO) ===")
    models["CQPPO"] = PPO("MlpPolicy", env_causal, policy_kwargs=policy_kwargs_quantum, learning_rate=1e-3, verbose=0)
    models["CQPPO"].learn(total_timesteps=total_timesteps)
    models["CQPPO"].save(os.path.join(output_dir, "CQPPO.zip"))

    print("=== 3. Training Standard Quantum-DQN (QDQN) ===")
    models["QDQN"] = DQN("MlpPolicy", env_standard, policy_kwargs=policy_kwargs_quantum, learning_rate=1e-3, verbose=0)
    models["QDQN"].learn(total_timesteps=total_timesteps)
    models["QDQN"].save(os.path.join(output_dir, "QDQN.zip"))

    print("=== 4. Training Classical-DQN Baseline ===")
    models["Classical-DQN"] = DQN("MlpPolicy", env_standard, learning_rate=1e-3, verbose=0)
    models["Classical-DQN"].learn(total_timesteps=total_timesteps)
    models["Classical-DQN"].save(os.path.join(output_dir, "Classical-DQN.zip"))

    print(f"\nAll models successfully trained and saved to '{output_dir}/'.")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal-QDRL IoT Agents")
    parser.add_argument("--timesteps", type=int, default=2000, help="Total training timesteps per agent")
    parser.add_argument("--data_path", type=str, default=None, help="Path to IoT dataset CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    train_causal_agents(total_timesteps=args.timesteps, data_path=args.data_path, output_dir=args.output_dir)
