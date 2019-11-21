import gym
from gym.envs.registration import register


from .my_frozen_lake import *
from .jack import *
from .hanoi import *
from .nchain import *
from .guess import *
from .simple import *
from .inventory import *

__all__ = ['MyFrozenLakeEnv','JacksCarRental','HanoiEnv','NChainEnv','GuessingGame','Simple','InventoryEnv']