import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.dqn import *

class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        device,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
        model_savefile="dqn_doom.pth",
        optimizer = 'Adam',
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.device = device


        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min
        else:
            print("Initializing New Model")
            self.q_net = DQN(action_size).to(device)

        if optimizer =='SGD':
            self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)
        elif optimizer == 'Adam':
            self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)


    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(self.device)
            next_state_values = torch.max(self.q_net(next_states), dim=1)[0]
            next_state_values = next_state_values.cpu().detach().numpy()
            next_state_values = next_state_values[not_dones]

        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values

        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(self.device)
        action_values = self.q_net(states)[idx].float().to(self.device)

        self.opt.zero_grad()
        td_error = self.criterion(torch.from_numpy(q_targets).float().to(self.device), action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
