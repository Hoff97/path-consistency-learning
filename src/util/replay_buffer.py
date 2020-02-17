import random

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.items = []
        self.capacity = capacity

    def insert(self, seq):
        if len(self.items) < self.capacity:
            self.items.append(seq)
        else:
            ix = random.randint(0,self.capacity - 1)
            self.items[ix] = seq

    def get_random(self):
        ix = random.randint(0,len(self.items) - 1)
        return self.items[ix]
