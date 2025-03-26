import sys
import os
sys.path.append(os.path.dirname(__file__))

from nn_model import NeuralNetworkModel
# other imports...
import sys
import os

# Add project root to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from agents.agent import Agent
from gym_env import PokerEnv
import random
from treys import Evaluator
import torch

action_types = PokerEnv.ActionType


class MonteCarloNNPlayer:
    def __init__(self, model_path=None):
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.model = NeuralNetworkModel()
        self.model.load(model_path)

    def act(self, state):
        features = self.extract_features(state)
        action_evs = self.model.predict(features)
        
        legal_actions = state.get_legal_actions()
        best_action = max(legal_actions, key=lambda a: action_evs[a])
        return best_action

    def extract_features(self, state):
        return [state.street, state.pot, state.player_stack]
    
class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0
        self.evaluator = Evaluator()

        # ✅ Load trained model with correct path
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.model = NeuralNetworkModel()
        self.model.load(model_path)

    def act(self, observation, reward, terminated, truncated, info):
        if observation["street"] == 0 and info.get("hand_number", 0) % 50 == 0:
            self.logger.info(f"Hand number: {info.get('hand_number', 'N/A')}")

        valid_actions = observation["valid_actions"]
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]

        # Extract features
        features = self.extract_features(observation, info)

        # Predict EVs for [fold, call, raise, discard, etc.]
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32)
            evs = self.model(input_tensor).numpy()

        # Choose best *valid* action
        best_action = max(valid_action_indices, key=lambda a: evs[a])

        # Default values
        raise_amount = 0
        card_to_discard = -1

        # If raising, pick a valid raise amount
        if best_action == action_types.RAISE.value:
            if observation["min_raise"] == observation["max_raise"]:
                raise_amount = observation["min_raise"]
            else:
                raise_amount = random.randint(observation["min_raise"], observation["max_raise"])

        # If discarding, randomly discard one of two cards
        if best_action == action_types.DISCARD.value:
            card_to_discard = random.randint(0, 1)

        return best_action, raise_amount, card_to_discard

    def extract_features(self, observation, info):
        pot = observation["my_bet"] + observation["opp_bet"]
        my_stack = 1000  # Mock/placeholder for now
        opp_stack = 1000
        bet_to_call = observation["opp_bet"] - observation["my_bet"]
        prev_action = 0  # or hardcode for now
        street = observation["street"]
        hand_strength = 0.0

        # Estimate hand strength only if board is visible
        if street > 0:
            try:
                my_cards = [PokerEnv.int_to_card(card) for card in observation["my_cards"] if card != -1]
                community_cards = [PokerEnv.int_to_card(card) for card in observation["community_cards"] if card != -1]
                hand_strength = 1 - self.evaluator.evaluate(my_cards, community_cards) / 7462
            except Exception as e:
                print("⚠️ Hand strength eval error:", e)
                hand_strength = 0.5  # fallback

        features = [
            float(pot),
            float(my_stack),
            float(opp_stack),
            float(bet_to_call),
            float(prev_action),
            float(hand_strength),
        ]

        # One-hot street
        one_hot = [0.0, 0.0, 0.0, 0.0]
        if 0 <= street < 4:
            one_hot[street] = 1.0
        features.extend(one_hot)

        return features
    

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")