# 导入必要的库
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q网络模型
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)

# DQN智能体
class Agent:
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 超参数设置
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率初始值
        self.epsilon_min = 0.05  # 探索率最小值
        self.epsilon_decay = 0.995  # 探索率衰减
        self.learning_rate = 1e-3  # 学习率
        self.batch_size = 128  # 批次大小
        self.memory_capacity = 1000000  # 经验回放容量
        self.target_update = 100  # 目标网络更新频率

        # 初始化经验回放缓冲区
        self.memory = ReplayMemory(self.memory_capacity)

        # 初始化Q网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.steps_done = 0

    # ε-贪婪策略选择动作
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    # 存储经验到回放缓冲区
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # 训练步骤
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放中采样
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # 准备批次数据
        states = torch.FloatTensor(batch[0])
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)

        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # 计算损失并更新网络
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 更新目标网络
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

