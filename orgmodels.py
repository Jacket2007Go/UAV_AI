import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, w_dim=2, hidden_dim=128):
        super().__init__()
        self.w_dim = w_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 + w_dim),
        )

    def forward(self, x):
        out = self.net(x)
        p = torch.sigmoid(out[..., :1])             # [0,1]
        w_raw = out[..., 1:]
        w = torch.tanh(w_raw)
        w = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-8)
        return torch.cat([p, w], dim=-1)


class CentralizedQCritic(nn.Module):
    """
    Centralized Q critic: Q(s_global, a_joint).
    s_global = concat of all obs, a_joint = concat of all actions.
    """
    def __init__(self, global_obs_dim, joint_act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + joint_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_global, a_joint):
        x = torch.cat([s_global, a_joint], dim=-1)
        return self.net(x)