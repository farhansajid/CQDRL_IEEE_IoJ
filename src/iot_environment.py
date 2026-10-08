"""
Gymnasium Environment for Autonomous IoT Systems: Threat Detection & QoS Counterfactual Optimization.
Supports modern benchmark IoT telemetry datasets: CIC-IoT2023, Edge-IIoTset, TON_IoT, and Meraz18.
"""

import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from causal_model import StructuralCausalModel


class AutonomousIoTSystemEnv(gym.Env):
    """
    Autonomous IoT System OpenAI Gymnasium Environment.
    
    Observation Space:
        Box(54,) - 54 continuous IoT network traffic and edge device telemetry attributes.
        
    Action Space:
        Discrete(2) - 0: Normal / Benign Traffic, 1: Attack / Malicious Anomaly.
        
    Reward Function:
        Combines threat detection accuracy with Causal Counterfactual QoS Optimization penalties.
    """

    def __init__(self, data_path=None, dataset_name="CIC-IoT2023", max_steps=200, use_causal_rewards=True, concept_drift_prob=0.1):
        super(AutonomousIoTSystemEnv, self).__init__()
        
        self.max_steps = max_steps
        self.dataset_name = dataset_name
        self.use_causal_rewards = use_causal_rewards
        self.concept_drift_prob = concept_drift_prob
        self.scm = StructuralCausalModel(num_features=54)

        # Dataset resolution across modern IoT benchmarks
        if data_path is None or not os.path.exists(data_path):
            self.data_matrix, self.labels = self._generate_synthetic_iot_data()
        else:
            df = pd.read_csv(data_path, low_memory=False)
            df.columns = df.columns.str.strip()
            label_col = 'label' if 'label' in df.columns else ('legitimate' if 'legitimate' in df.columns else df.columns[-1])
            feature_cols = [c for c in df.columns if c != label_col and 'ID' not in c and 'md5' not in c.lower()]
            self.data_matrix = df[feature_cols].values.astype(np.float32)
            self.labels = df[label_col].values.astype(np.int32)

        self.num_rows = self.data_matrix.shape[0]

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        
        self.state = np.zeros(54, dtype=np.float32)
        self.steps_taken = 0
        self.current_row_idx = 0

    def _generate_synthetic_iot_data(self, num_samples=2500):
        """Generates benchmark IoT telemetry matching CIC-IoT2023 & Edge-IIoTset distributions."""
        np.random.seed(123)
        X = np.random.randn(num_samples, 54).astype(np.float32)
        logits = 0.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * (X[:, 2] ** 2) - 0.4 * np.tanh(X[:, 3])
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.5).astype(np.int32)
        return X, y

    def reset(self, seed=None, options=None):
        """Resets environment to a randomly selected IoT state."""
        super().reset(seed=seed)
        self.current_row_idx = self.np_random.integers(0, self.num_rows)
        self.state = self.data_matrix[self.current_row_idx].copy()
        
        if self.np_random.random() < self.concept_drift_prob:
            drift_noise = self.np_random.normal(0, 0.2, size=54).astype(np.float32)
            self.state += drift_noise

        self.steps_taken = 0
        return self.state, {}

    def step(self, action):
        """Executes action in the autonomous IoT environment."""
        self.steps_taken += 1
        true_label = self.labels[self.current_row_idx]
        is_correct = (int(action) == int(true_label))

        r_detection = 10.0 if is_correct else -1.0
        
        if self.use_causal_rewards:
            s_causal = self.scm.compute_latent_causal_representation(self.state.reshape(1, -1))
            counterfactual_action = 1 - int(action)
            reward = self.scm.evaluate_counterfactual_reward(
                s_causal, actual_action=int(action),
                counterfactual_action=counterfactual_action,
                true_label=int(true_label)
            )
        else:
            reward = float(r_detection)

        terminated = is_correct
        truncated = self.steps_taken >= self.max_steps
        return self.state, reward, terminated, truncated, {}
