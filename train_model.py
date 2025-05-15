import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from data_processing import read_data
from model_definition import DQN

# 读取数据
worker_quality, project_info, entry_info = read_data()

# 定义超参数
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 128
TARGET_UPDATE = 10

# 初始化模型和目标模型
state_dim = 5  # 确保和测试时的输入维度一致
action_dim = 10  # 确保和测试时的输出维度一致
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 定义优化器和损失函数
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.SmoothL1Loss()

# 定义经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)

# 定义选择动作的函数
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)

# 定义训练函数
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()  # 确保形状一致

    # 确保形状一致
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # 打印形状信息，用于调试
    print(f"state_action_values shape: {state_action_values.shape}")
    print(f"expected_state_action_values shape: {expected_state_action_values.shape}")

    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# 训练循环
num_episodes = 50
for i_episode in range(num_episodes):
    # 随机选择一个工人
    worker_id = random.choice(list(worker_quality.keys()))

    # 初始化状态
    state = torch.randn(1, state_dim)  # 确保状态维度正确

    for t in range(100):
        action = select_action(state)

        # 执行动作，获取奖励和下一个状态
        reward = torch.tensor([[random.random()]], dtype=torch.float)
        next_state = torch.randn(1, state_dim)  # 确保状态维度正确

        memory.push(state, action, reward, next_state)

        state = next_state

        optimize_model()

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Training complete')
torch.save(policy_net.state_dict(), 'dqn_model.pth')