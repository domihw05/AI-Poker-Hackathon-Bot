import torch
import torch.optim as optim
from nn_model import NeuralNetworkModel
from monte_carlo import monte_carlo_eval
from player import PlayerAgent

import random
import time
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from gym_env import PokerEnv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../agents"))
sys.path.append(root_dir)
from test_agents import AllInAgent  # or whatever path AllInAgent is under
from test_agents import ChallengeAgent

# Placeholder classes
class GameState:
    def __init__(self, street=0, pot=0, player_stack=0):
        self.street = street  # 0=preflop, 1=flop, 2=turn, 3=river
        self.pot = pot
        self.player_stack = player_stack

        # 🔧 Add these new attributes for feature extraction:
        self.opponent_stack = random.randint(100, 1000)
        self.bet_to_call = random.randint(0, 100)
        self.previous_action = random.randint(0, 2)  # 0=check, 1=call, 2=raise
        self.hand_strength = random.uniform(0, 1)    # Mocked win probability

        self.terminal = False
        self.steps_taken = 0

    def get_legal_actions(self):
        return [0, 1, 2]  # fold, call, raise

    def is_terminal(self):
        return self.terminal or self.steps_taken >= 5  # auto-end after 5 steps

    def apply_action(self, action):
        self.steps_taken += 1
        # Placeholder: could update pot, stack, etc.

    def current_player(self):
        return "player"

    def get_player_reward(self, player_id):
        return random.uniform(-1, 1)

    def clone(self):
        from copy import deepcopy
        return deepcopy(self)



def extract_features(state):
    features = [
        state.pot,
        state.player_stack,
        state.opponent_stack,
        state.bet_to_call,
        state.previous_action,
        state.hand_strength
    ]

    # One-hot encode street (4 values)
    street_one_hot = [0, 0, 0, 0]
    street_one_hot[state.street] = 1
    features.extend(street_one_hot)

    return features  # Total = 6 + 4 = 10

def train_model(num_episodes=10, n_simulations=10):
    model = NeuralNetworkModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    env = PokerEnv()
    start_time = time.time()

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        state = env.reset()

        step = 0
        step = 0
        max_steps_per_episode = 20  # Safety cap to prevent infinite loops
        while not state.is_terminal() and step < max_steps_per_episode:
            legal_actions = state.get_legal_actions()
            feature_vector = extract_features(state)
            
            target_evs = [0.0] * 5  # initialize all EVs to 0.0

            for action in legal_actions:
                ev = monte_carlo_eval(state, action, "player", n_simulations=n_simulations)
                target_evs[action] = ev
                print(f"  Step {step}, Action {action}: EV={ev:.4f}")

            prediction = model(torch.tensor(feature_vector).float())
            target_tensor = torch.tensor(target_evs, dtype=torch.float32)

            loss = loss_fn(prediction, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    model.save("model.pth")

def collect_training_data(num_hands_per_opponent=250):
    opponents = {
        "ChallengeAgent": ChallengeAgent,
        "AllInAgent": AllInAgent
    }

    data = []

    for label, OpponentClass in opponents.items():
        print(f"🔄 Collecting hands vs {label}...")

        for _ in range(num_hands_per_opponent):
            env = PokerEnv()
            agent0 = PlayerAgent(stream=False)
            agent1 = OpponentClass(stream=False)

            obs, _ = env.reset()
            terminated = False
            truncated = False
            reward = (0, 0)
            info = {}

            states = []
            actions = []

            while not terminated:
                curr_agent = agent0 if env.acting_agent == 0 else agent1
                obs_self = obs[env.acting_agent]
                rew = reward[env.acting_agent]

                action = curr_agent.act(obs_self, rew, terminated, truncated, info)

                if isinstance(curr_agent, PlayerAgent):
                    features = agent0.extract_features(obs_self, info)
                    states.append(features)
                    actions.append(action[0])

                obs, reward, terminated, truncated, info = env.step(action)

            discount_factor = 0.9  # can tune this

            # After the hand ends
            final_reward = reward[0]
            for step_index, (f, a) in enumerate(zip(states, actions)):
                discounted = final_reward * (discount_factor ** (len(states) - step_index - 1))
                data.append((f, a, discounted))

    return data

def train_on_match_data(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for features, action_taken, reward in data:
        input_tensor = torch.tensor(features, dtype=torch.float32)
        target = torch.zeros(5)
        target[action_taken] = reward  # only one action got the reward

        prediction = model(input_tensor)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    data = collect_training_data(num_hands_per_opponent=10000)
    model = NeuralNetworkModel()
    train_on_match_data(data, model)
    model.save("submission/model.pth")