"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.interaction_buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add_local_critic(
        self,
        s,
        cs,
        a,
        r,
        t,
        s2,
        cs2,
        interaction=False,
    ):
        experience = (s, cs, a, r, t, s2, cs2, bool(interaction))
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            expired = self.buffer.popleft()
            if len(expired) >= 8 and expired[7]:
                self.interaction_buffer.popleft()
            self.buffer.append(experience)
        if interaction:
            self.interaction_buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def sample_local_critic_batch(
        self, batch_size, interaction_only=False, interaction_fraction=0.0
    ):
        if interaction_fraction < 0.0 or interaction_fraction > 1.0:
            raise ValueError("interaction_fraction must be in [0, 1]")
        source = self.interaction_buffer if interaction_only else self.buffer
        if not source:
            return None

        if interaction_only or interaction_fraction <= 0.0:
            count = len(source)
            batch = random.sample(source, min(count, batch_size))
        else:
            total_count = min(len(self.buffer), batch_size)
            interaction_count = min(
                len(self.interaction_buffer),
                int(round(total_count * interaction_fraction)),
            )
            batch = random.sample(self.interaction_buffer, interaction_count)
            non_interaction = [
                experience
                for experience in self.buffer
                if len(experience) < 8 or not experience[7]
            ]
            non_interaction_count = min(
                len(non_interaction), total_count - len(batch)
            )
            batch.extend(
                random.sample(non_interaction, non_interaction_count)
            )
            if len(batch) < total_count:
                selected_ids = {id(experience) for experience in batch}
                fallback = [
                    experience
                    for experience in self.buffer
                    if id(experience) not in selected_ids
                ]
                batch.extend(
                    random.sample(fallback, total_count - len(batch))
                )
            random.shuffle(batch)

        s_batch = np.array([_[0] for _ in batch])
        cs_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        r_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[4] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[5] for _ in batch])
        cs2_batch = np.array([_[6] for _ in batch])

        return s_batch, cs_batch, a_batch, r_batch, t_batch, s2_batch, cs2_batch

    def clear(self):
        self.buffer.clear()
        self.interaction_buffer.clear()
        self.count = 0

    def interaction_size(self):
        return len(self.interaction_buffer)

    def state_dict(self):
        return {
            "buffer_size": self.buffer_size,
            "count": self.count,
            "buffer": list(self.buffer),
            "interaction_buffer": list(self.interaction_buffer),
        }

    def load_state_dict(self, state):
        self.buffer_size = state["buffer_size"]
        self.count = state["count"]
        self.buffer = deque(state["buffer"], maxlen=None)
        stored_interactions = state.get("interaction_buffer")
        if stored_interactions is None:
            stored_interactions = [
                item for item in self.buffer if len(item) >= 8 and item[7]
            ]
        self.interaction_buffer = deque(stored_interactions, maxlen=None)
