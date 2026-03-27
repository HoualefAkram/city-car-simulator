import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Custom Imports ---
from data_models.latlng import LatLng
from handover_env import HandoverEnv
from replay_buffer import ReplayBuffer
from checkpoint_manager import CheckpointManager

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 256),  # 12 continuous states (normalized floats)
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(
                64, 4
            ),  # 4 outputs representing Q-values for the 4 candidate towers
        )

    def forward(self, x):
        return self.net(x)


def hard_update(target_net, policy_net):
    target_net.load_state_dict(policy_net.state_dict())


# ==========================================
# 2. INITIALIZATION & HYPERPARAMETERS
# ==========================================

# Initialize the environment
MAP_TOP_LEFT = LatLng(51.519411, -0.148076)  # London
MAP_BOTTOM_RIGHT = LatLng(51.499324, -0.109732)  # London
MCC = 234  # UK
env = HandoverEnv(top_left=MAP_TOP_LEFT, bottom_right=MAP_BOTTOM_RIGHT, mcc=MCC)

epoches = 500
lr = 1e-3
decay_val = 0.99995
min_epsilon = 0.05
gamma = 0.99
update_rate = 100
batch_size = 32

policy_network = QNetwork()
target_network = QNetwork()
hard_update(target_network, policy_network)

criterion = nn.MSELoss()
adam = optim.Adam(policy_network.parameters(), lr=lr)

# Initialize external utility classes
memory = ReplayBuffer(file_path="training/replay_buffer.pkl", max_len=10000)
checkpoint_manager = CheckpointManager(file_path="training/ddqn_checkpoint.pth")

# Load existing training state if it exists!
start_epoch, epsilon = checkpoint_manager.load_checkpoint(
    policy_network, target_network, adam, default_epsilon=1.0
)


# ==========================================
# 3. TRAINING LOOP
# ==========================================

target_net_update_counter = 0

print("--- Starting DDQN Training ---")
for epoche in range(start_epoch, epoches):
    done = False
    state, _ = env.reset()
    total_reward = 0
    step_count = 0

    while not done:
        # Convert state (numpy array) to PyTorch tensor [1, 12]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(policy_network(state_tensor)).item()

        # Step Environment
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # Save to RAM buffer
        memory.append((state, action, reward, new_state, done))
        state = new_state

        # Train the network (Double DQN)
        if len(memory) >= batch_size:
            batch = random.sample(memory.queue, batch_size)
            b_states, b_actions, b_rewards, b_new_states, b_dones = zip(*batch)

            # Efficient tensor conversion
            b_states_t = torch.tensor(np.array(b_states), dtype=torch.float32)
            b_new_states_t = torch.tensor(np.array(b_new_states), dtype=torch.float32)
            b_actions_t = torch.tensor(b_actions, dtype=torch.int64).unsqueeze(1)
            b_rewards_t = torch.tensor(b_rewards, dtype=torch.float32).unsqueeze(1)
            b_dones_t = torch.tensor(b_dones, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                # Policy net evaluates WHICH action is best
                best_next_action_idxs = torch.argmax(
                    policy_network(b_new_states_t), dim=1, keepdim=True
                )
                # Target net evaluates the Q-VALUE of that chosen action
                target_optimal_next_qs = target_network(b_new_states_t)
                v_targets = target_optimal_next_qs.gather(1, best_next_action_idxs)

                # Bellman equation
                bellman_targets = b_rewards_t + gamma * v_targets * (1 - b_dones_t)

            # Get current Q-value predictions
            policy_preds = policy_network(b_states_t).gather(1, b_actions_t)

            loss = criterion(policy_preds, bellman_targets)

            adam.zero_grad()
            loss.backward()
            adam.step()

            target_net_update_counter += 1
            if target_net_update_counter >= update_rate:
                target_net_update_counter = 0
                hard_update(target_network, policy_network)

    # Decay Epsilon at the end of the episode
    epsilon = max(min_epsilon, epsilon * decay_val)

    # Freeze-Dry (Save) the buffer and the neural networks
    memory.save()
    checkpoint_manager.save_checkpoint(
        epoche, epsilon, policy_network, target_network, adam
    )

    print(
        f"Epoch {epoche+1}/{epoches} | Steps: {step_count} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
    )
