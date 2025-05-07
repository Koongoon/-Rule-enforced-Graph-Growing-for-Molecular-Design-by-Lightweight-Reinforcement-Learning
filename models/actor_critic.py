import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

class ActorCritic(nn.Module):
    def __init__(self, node_embedding_size=32, num_node_types=6):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(node_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.node_type_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_node_types)
        )

        self.continue_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.critic = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, node_embedding):
        actor_features = self.actor(node_embedding)
        node_type_logits = self.node_type_head(actor_features)
        node_type_probs = F.softmax(node_type_logits, dim=-1)

        continue_logits = self.continue_head(actor_features)
        continue_prob = torch.sigmoid(continue_logits)

        state_value = self.critic(actor_features)

        return node_type_probs, continue_prob, state_value
