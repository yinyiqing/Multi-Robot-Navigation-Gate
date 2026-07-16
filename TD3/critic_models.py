import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, state, action):
        state_1 = F.relu(self.layer_1(state))
        state_1 = F.linear(state_1, self.layer_2_s.weight)
        action_1 = self.layer_2_a(action)
        q1 = self.layer_3(F.relu(state_1 + action_1))

        state_2 = F.relu(self.layer_4(state))
        state_2 = F.linear(state_2, self.layer_5_s.weight)
        action_2 = self.layer_5_a(action)
        q2 = self.layer_6(F.relu(state_2 + action_2))
        return q1, q2
