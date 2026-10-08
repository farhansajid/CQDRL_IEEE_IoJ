"""
Structural Causal Model (SCM), Causal Discovery, and Counterfactual Reasoning Module for Autonomous IoT Systems.
Implements Judea Pearl's Do-Calculus interventions, PC Causal Discovery simulation, and Counterfactual Reward Optimization.
"""

import numpy as np
import networkx as nx


class StructuralCausalModel:
    """
    Structural Causal Model (SCM) for Autonomous IoT Telemetry & Threat Dynamics.
    
    Causal Graph DAG:
        - X (IoT Attributes: CPU, Memory, Packet Rate, Jitter, Section Entropy)
        - U (Unobserved Noise / Environmental Heterogeneity)
        - S (System Causal Latent State) -> S = f_S(X, U)
        - A (Intervention Action / Threat Countermeasure) -> A = f_A(S)
        - Q (QoS Metrics: Latency, Energy, Bandwidth) -> Q = f_Q(S, A, U_Q)
        - Y (Security Outcome / Threat Detection) -> Y = f_Y(S, A, Q, U_Y)
    """

    def __init__(self, num_features=54):
        self.num_features = num_features
        self.dag = self._build_causal_graph()
        
        # Structural Equation Matrices
        np.random.seed(42)
        self.W_xs = np.random.randn(num_features, 4) * 0.1
        self.W_sq = np.random.randn(4, 3) * 0.1  # QoS: Latency, Energy, Bandwidth
        self.W_sy = np.random.randn(4, 1) * 0.1
        self.W_aq = np.random.randn(1, 3) * 0.2
        self.W_ay = np.random.randn(1, 1) * 0.5

    def _build_causal_graph(self):
        """Constructs the Directed Acyclic Graph (DAG) for the IoT system."""
        G = nx.DiGraph()
        G.add_edges_from([
            ("IoT_Attributes", "Causal_Latent_State"),
            ("Causal_Latent_State", "QoS_Metrics"),
            ("Causal_Latent_State", "Security_Outcome"),
            ("Action_Intervention", "QoS_Metrics"),
            ("Action_Intervention", "Security_Outcome"),
            ("QoS_Metrics", "Security_Outcome")
        ])
        return G

    def discover_causal_graph(self, x_samples):
        """Simulates PC Algorithm constraint-based causal discovery from observational data."""
        cov_matrix = np.cov(x_samples.T)
        edges = []
        for i in range(min(5, x_samples.shape[1])):
            for j in range(i+1, min(5, x_samples.shape[1])):
                if abs(cov_matrix[i, j]) > 0.1:
                    edges.append((f"Feature_{i}", f"Feature_{j}"))
        return edges

    def compute_latent_causal_representation(self, x_obs):
        """Maps high-dimensional IoT telemetry into a causal latent space S."""
        u = np.random.normal(0, 0.05, size=(x_obs.shape[0], 4))
        s_causal = np.tanh(np.dot(x_obs, self.W_xs) + u)
        return s_causal

    def do_intervention(self, s_causal, action_val):
        """
        Executes Pearl's do-calculus intervention: do(A = action_val).
        Evaluates causal effects on QoS metrics (Latency in ms, Energy in Joules, Bandwidth in Mbps).
        """
        a_vec = np.full((s_causal.shape[0], 1), action_val, dtype=np.float32)
        qos_metrics = np.tanh(np.dot(s_causal, self.W_sq) + np.dot(a_vec, self.W_aq))
        return qos_metrics

    def evaluate_counterfactual_reward(self, s_causal, actual_action, counterfactual_action, true_label):
        """
        Computes Counterfactual Reward:
        'What would the QoS penalty and detection confidence have been if action a' had been executed instead of a?'
        """
        # Actual QoS & Security Outcome
        qos_actual = self.do_intervention(s_causal, actual_action)
        is_actual_correct = (actual_action == true_label)
        r_actual = 10.0 if is_actual_correct else -1.0
        
        # Counterfactual QoS under do(A = a')
        qos_cf = self.do_intervention(s_causal, counterfactual_action)
        is_cf_correct = (counterfactual_action == true_label)
        r_cf = 10.0 if is_cf_correct else -1.0

        # Causal Regret / Counterfactual Advantage
        qos_latency_penalty = np.mean(np.abs(qos_actual[:, 0] - qos_cf[:, 0]))
        r_counterfactual = r_actual - 0.2 * qos_latency_penalty + 0.1 * (r_cf - r_actual)

        return float(r_counterfactual)
