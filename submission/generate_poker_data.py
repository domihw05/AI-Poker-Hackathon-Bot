import pandas as pd
import torch
from torch.utils.data import Dataset

# Action space
ACTIONS = ['FOLD', 'CALL', 'RAISE', 'CHECK', 'DISCARD']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Encode cards
def encode_card(card):
    suits = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
             '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    return ranks[card[0]] * 4 + suits[card[1]]

def encode_cards(cards, max_len):
    vec = [0] * (max_len * 52)
    for i, card in enumerate(cards[:max_len]):
        if len(card) == 2:
            idx = encode_card(card)
            vec[i * 52 + idx] = 1
    return vec

# Encode each row into a feature vector
def encode_row(row):
    features = []

    # Encode street
    streets = ['Pre-Flop', 'Flop', 'Turn', 'River']
    street_onehot = [1 if row['street'] == s else 0 for s in streets]
    features += street_onehot

    # Encode cards
    team_0_cards = eval(row['team_0_cards']) if isinstance(row['team_0_cards'], str) else []
    team_1_cards = eval(row['team_1_cards']) if isinstance(row['team_1_cards'], str) else []
    board = eval(row['board_cards']) if isinstance(row['board_cards'], str) else []

    player_cards = team_1_cards if int(row['active_team']) == 1 else team_0_cards
    features += encode_cards(player_cards, 2)
    features += encode_cards(board, 5)

    # Add bankrolls and bets
    for col in ['team_0_bankroll', 'team_1_bankroll', 'team_0_bet', 'team_1_bet']:
        try:
            features.append(float(row[col]))
        except:
            features.append(0.0)

    return features

# Get action label
def extract_action(row):
    return ACTION_TO_IDX.get(row['action_type'], None)

# Dataset class
class PokerDataset(Dataset):
    def __init__(self, features, actions):
        self.inputs = features
        self.labels = actions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Load CSV and process rows
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    features, actions = [], []

    for _, row in df.iterrows():
        action = extract_action(row)
        if action is not None:
            feat = encode_row(row)
            features.append(feat)
            actions.append(action)

    return {'features': features, 'actions': actions}

# MAIN
if __name__ == "__main__":
    data = load_data_from_csv("../match.csv")  # <- path to your CSV
    print(f"Total samples: {len(data['features'])}")
    print(f"Input dimension: {len(data['features'][0])}")
    print(f"Example feature vector:\n{data['features'][0]}")
    print(f"Example action (label): {data['actions'][0]}")

    # Optionally, save to disk
    torch.save(data, "poker_data.pt")