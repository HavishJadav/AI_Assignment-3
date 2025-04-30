# agents/alphabeta_agent.py

import numpy as np
from utils.evaluation import evaluate as slime_evaluate


class SlimeVolleyAlphaBetaAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def act(self, obs):
        _, best_action = self.alphabeta(obs, self.depth, -np.inf, np.inf, True)
        return self.decode_discrete_action(best_action)

    @staticmethod
    def decode_discrete_action(action_idx):
        return [
            int(action_idx in [1, 3, 5, 7]),
            int(action_idx in [2, 3, 6, 7]),
            int(action_idx in [4, 5, 6, 7])
        ]

    def alphabeta(self, state, depth, alpha, beta, maximizing):
        if depth == 0:
            return slime_evaluate(state), None

        legal_actions = list(range(8))  # 8 discrete actions
        best_action = None

        if maximizing:
            max_eval = -np.inf
            for action in legal_actions:
                new_state = self.simulate(state, action)
                eval_score, _ = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = np.inf
            for action in legal_actions:
                new_state = self.simulate(state, action)
                eval_score, _ = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def simulate(self, state, action):
        # Placeholder for forward model simulation
        return state
