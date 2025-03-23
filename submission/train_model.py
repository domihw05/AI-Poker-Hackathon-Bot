import torch
import torch.optim as optim
from model import PokerPolicyNet

# Create network
input_dim = 20  # depends on your feature vector
output_dim = 5  # fold, raise, check, call, discard
policy = PokerPolicyNet(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# Simulate games here...
for episode in range(num_episodes):
    log_probs = []
    rewards = []

    obs, _ = env.reset()
    done = False
    while not done:
        features = build_feature_vector(obs)
        features_tensor = torch.tensor(features).float().unsqueeze(0)

        logits = policy(features_tensor)
        mask = torch.tensor(valid_action_mask).bool()
        logits[~mask] = -1e9  # mask invalid actions

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, _, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)

        obs = next_obs

    # Calculate return and update policy
    total_reward = sum(rewards)
    loss = -sum([lp * total_reward for lp in log_probs])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()