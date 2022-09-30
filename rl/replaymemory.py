import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('g', 'a', 'mask', 'r', 'ng', 'm_mask', 't'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = Transition

    def push(self, *args):
        if args[0] is None:
            return
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size, on_policy=False):
        if on_policy:
            samples = self.memory

        else:
            samples = random.sample(self.memory, batch_size)

        return list(zip(*samples))

    def __len__(self):
        return len(self.memory)
