import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, card_list=None):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size
        self.card_list = card_list if card_list else ["card0","card1","card2","card3"]

        # Initialize counter-learning table
        # Format: {enemy_type: [score_for_card0, score_for_card1, card2, card3]}
        self.counter_table = {}

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done, enemy_classes=None, card_index=None):
        """
        Stores the transition and updates counter-learning table if reward is positive
        """
        self.memory.append((s, a, r, s2, done, enemy_classes, card_index))
        if enemy_classes and card_index is not None and r > 0:
            for e_type in enemy_classes:
                if e_type not in self.counter_table:
                    self.counter_table[e_type] = np.zeros(len(self.card_list))
                self.counter_table[e_type][card_index] += r  # accumulate reward for card vs enemy

    def act(self, state, enemy_classes=None):
        """
        Select an action, biased by counter table if enemies detected
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_index = q_values.argmax().item()

        # Adjust action based on counters
        if enemy_classes:
            best_card = None
            best_score = -np.inf
            for e_type in enemy_classes:
                if e_type in self.counter_table:
                    scores = self.counter_table[e_type]
                    top_card_idx = np.argmax(scores)
                    if scores[top_card_idx] > best_score:
                        best_score = scores[top_card_idx]
                        best_card = top_card_idx
            if best_card is not None:
                # choose a grid position for this card dynamically
                grid_x = random.randint(0, 17)
                grid_y = random.randint(0, 27)
                action_index = best_card * 18 * 28 + grid_x * 28 + grid_y

        return action_index

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, enemy_classes, card_index in batch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state)))
            target_f = self.model(torch.FloatTensor(state))
            target_f = target_f.clone()
            target_f[action] = float(target)

            prediction = self.model(torch.FloatTensor(state))[action]
            loss = self.criterion(prediction, target_f[action].detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Loaded model weights from {path}")

    def save(self, filename):
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model weights to {path}")
