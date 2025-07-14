import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Define Policy and Value Networks 
class PolicyNetwork(nn.Module):
    """
    Represents the 'actor' in PPO. Takes a state and outputs action probabilities.
    For an LLM, this would be the LLM itself, taking a prompt/context and
    outputting probabilities over the next token.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        # Use softmax for discrete action probabilities
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    """
    Represents the 'critic' in PPO. Takes a state and outputs its estimated value.
    For an LLM, this would be a separate network estimating the 'goodness' of a state (prompt/context).
    """
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1) # Output a single value

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Generalized Advantage Estimation (GAE) Function
def compute_gae(rewards, values, dones, gamma, lambda_):
    """
    Computes Generalized Advantage Estimation (GAE).
    Args:
        rewards (list/np.array): Rewards collected in the trajectory.
        values (list/np.array): Value estimates for each state in the trajectory.
        dones (list/np.array): Boolean indicating if episode ended at that step.
        gamma (float): Discount factor.
        lambda_ (float): GAE parameter for bias-variance trade-off.
    Returns:
        np.array: Computed advantages.
    """
    advantages = []
    # Convert to tensors for calculations
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Calculate TD errors (deltas)
    # V(s_t+1) for the last step is 0 if done, otherwise bootstrapped from next value
    next_values = torch.cat((values[1:], torch.tensor([0.0]))) # Append 0 for the last state
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # Calculate GAE advantages backwards
    gae = 0
    for delta, done in zip(reversed(deltas), reversed(dones)):
        gae = delta + gamma * lambda_ * gae * (1 - done) # (1 - done) ensures reset at episode end
        advantages.insert(0, gae) # Insert at beginning to maintain original order

    return torch.tensor(advantages, dtype=torch.float32)

# PPO Update Function 
def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value,
               states, actions, old_log_probs, returns, advantages,
               clip_epsilon, ppo_epochs, mini_batch_size):
    """
    Performs PPO updates on the policy and value networks.
    """
    # Convert data to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # Normalize advantages (important for stable training)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    data_size = states.size(0)
    indices = np.arange(data_size)

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, data_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]

            # Policy Loss
            new_action_probs = policy_net(batch_states)
            new_log_probs = Categorical(new_action_probs).log_prob(batch_actions)
            
            # Ratio of new policy to old policy probabilities
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() # Maximize objective -> minimize negative objective

            # Value Loss
            predicted_values = value_net(batch_states).squeeze(-1)
            value_loss = F.mse_loss(predicted_values, batch_returns)

            # Update Policy Network
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Update Value Network
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

# Simulation Environment (Dummy for demonstration) 
# In a real LLM setting, this would be the LLM generating text and a Reward Model scoring it.
class DummyEnvironment:
    def __init__(self, obs_dim, action_dim):
        self.observation_space_dim = obs_dim
        self.action_space_dim = action_dim
        self.current_state = np.random.rand(obs_dim)
        self.steps_taken = 0
        self.max_steps = 100

    def reset(self):
        self.current_state = np.random.rand(self.observation_space_dim)
        self.steps_taken = 0
        return self.current_state

    def step(self, action):
        self.steps_taken += 1
        # Simulate state transition (random for dummy env)
        next_state = self.current_state + np.random.randn(self.observation_space_dim) * 0.1
        # Simulate reward (random for dummy env, or based on action)
        reward = 1.0 if np.random.rand() > 0.5 else -1.0 # Simple binary reward
        done = self.steps_taken >= self.max_steps
        self.current_state = next_state
        return next_state, reward, done, {} # obs, reward, done, info

# Main Training Loop for PPO 
if __name__ == "__main__":
    import torch.nn.functional as F # Ensure F is imported for relu and softmax

    # Hyperparameters
    OBS_DIM = 4  # Example: CartPole state space
    ACTION_DIM = 2 # Example: CartPole action space (left/right)
    LR_POLICY = 3e-4
    LR_VALUE = 1e-3
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    PPO_EPOCHS = 10
    MINI_BATCH_SIZE = 64
    NUM_EPISODES = 500

    env = DummyEnvironment(OBS_DIM, ACTION_DIM)
    policy_net = PolicyNetwork(OBS_DIM, ACTION_DIM)
    value_net = ValueNetwork(OBS_DIM)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LR_POLICY)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LR_VALUE)

    print("Starting PPO Training Simulation...")
    for episode in range(NUM_EPISODES):
        states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []
        
        obs = env.reset()
        done = False
        episode_reward = 0

        # Collect trajectory
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # Add batch dim
            
            # Get action probabilities from policy
            action_probs = policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Get value estimate from value network
            value = value_net(state_tensor).item()

            next_obs, reward, done, _ = env.step(action.item())

            states.append(obs)
            actions.append(action.item())
            old_log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value)
            dones.append(done) # True/False

            obs = next_obs
            episode_reward += reward

        # Compute returns (discounted sum of rewards)
        # This is a basic return calculation, GAE is for advantages
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        
        # Compute GAE advantages
        advantages = compute_gae(rewards, values, dones, GAMMA, LAMBDA)

        # Perform PPO update
        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value,
                   states, actions, old_log_probs, returns, advantages,
                   CLIP_EPSILON, PPO_EPOCHS, MINI_BATCH_SIZE)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Total Reward: {episode_reward:.2f}")

    print("PPO Training Simulation Complete.")
