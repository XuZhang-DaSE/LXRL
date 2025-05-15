import torch
import random  # 导入random模块
import numpy as np
from model_definition import DQN
from dateutil.parser import parse  # 导入parse函数
from data_processing import read_data  # 导入read_data函数

# 读取数据
worker_quality, project_info, entry_info = read_data()

# 初始化模型
state_dim = 5  # 确保和训练时的输入维度一致
action_dim = 10  # 确保和训练时的输出维度一致
model = DQN(state_dim, action_dim)

# 加载模型状态字典
model.load_state_dict(torch.load("dqn_model.pth", weights_only=True))
model.eval()

def get_state(worker_id, project_id):
    # 根据实际情况构造状态
    state = [worker_quality.get(worker_id, 0),
             project_info[project_id]["sub_category"],
             project_info[project_id]["category"],
             (project_info[project_id]["start_date"] - parse("2018-01-01T0:0:0Z")).total_seconds(),
             (project_info[project_id]["deadline"] - parse("2018-01-01T0:0:0Z")).total_seconds()]
    return np.array(state, dtype=np.float32)

def get_reward(worker_id, project_id):
    # 简单示例：奖励为工人质量和项目回答数量的乘积
    worker_qual = worker_quality.get(worker_id, 0)
    entry_count = project_info[project_id]["entry_count"]
    return worker_qual * entry_count

total_reward = 0
test_episodes = 100

for episode in range(test_episodes):
    # 随机选择一个工人
    worker_id = random.choice(list(worker_quality.keys()))
    # 随机选择一个项目
    project_id = random.choice(list(project_info.keys()))

    state = get_state(worker_id, project_id)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = model(state_tensor)
    action = torch.argmax(q_values, dim=1).item()

    next_project_id = list(project_info.keys())[action]
    reward = get_reward(worker_id, next_project_id)
    total_reward += reward

    print(f"Episode: {episode}, Reward: {reward}")

average_reward = total_reward / test_episodes
print(f"Average Reward: {average_reward}")