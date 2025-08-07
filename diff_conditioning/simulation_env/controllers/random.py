from .base import Base

from softzoo.envs.base_env import BaseEnv

class Random(Base):
    def __init__(self, env:BaseEnv):
        super(Random, self).__init__()
        self.env = env

    def forward(self, s, inp):
        return self.env.action_space.sample()

    def update(self, grad, retain_graph=False):
        pass
