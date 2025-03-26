import random
from copy import deepcopy

def monte_carlo_eval(base_state, action, player_id, n_simulations=100):
    """
    Simulate N hands from the given state and estimate expected reward of an action.
    """
    print(f"    Simulating action {action} ({n_simulations} sims)...")
    total_reward = 0
    for _ in range(n_simulations):
        sim_state = base_state.clone()
        sim_state.apply_action(action)

        max_steps = 10
        sim_steps = 0

        while not sim_state.is_terminal() and sim_steps < max_steps:
            legal_actions = sim_state.get_legal_actions()
            sim_state.apply_action(random.choice(legal_actions))
            sim_steps += 1

        total_reward += sim_state.get_player_reward(player_id)

    return total_reward / n_simulations