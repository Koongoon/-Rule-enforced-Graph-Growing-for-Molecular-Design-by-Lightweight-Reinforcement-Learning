import random
from collections import deque
import torch

class ExperienceReplayBuffer:
    def __init__(self, max_size=100000):
        self.positive_buffer = deque(maxlen=max_size // 2)
        self.negative_buffer = deque(maxlen=max_size // 2)

    def add(self, state, action, continue_gen, reward, next_state, done, is_positive):
        state = state.detach()
        if next_state is not None:
            next_state = next_state.detach()
        if is_positive:
            self.positive_buffer.append((state, action, continue_gen, reward, next_state, done))
        else:
            self.negative_buffer.append((state, action, continue_gen, reward, next_state, done))

    def sample(self, batch_size):
        pos_size = min(len(self.positive_buffer), batch_size // 2)
        neg_size = min(len(self.negative_buffer), batch_size - pos_size)
        positive_samples = random.sample(self.positive_buffer, pos_size)
        negative_samples = random.sample(self.negative_buffer, neg_size)
        return positive_samples + negative_samples

    def __len__(self):
        return len(self.positive_buffer) + len(self.negative_buffer)
