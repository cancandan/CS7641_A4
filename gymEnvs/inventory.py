import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    TO BE EDITED

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, n=5, k=5, c=2, h=2, p=3, lam=1):
        self.n = n
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Discrete(n)
        self.max = n-1
        self.state = n-1
        self.k = k
        self.c = c
        self.h = h
        self.p = p
        self.lam = lam

        # Set seed
        self._seed()

        # Start the first round
        self._reset()

        self.nS=5
        self.nA = 5
        self.P = np.array([[[0.200115, 0.      , 0.      , 0.      , 0.      ],
                            [0.126604, 0.073648, 0.      , 0.      , 0.      ],
                            [0.052805, 0.073501, 0.074173, 0.      , 0.      ],
                            [0.016055, 0.036385, 0.073421, 0.073552, 0.      ],
                            [0.00388 , 0.01226 , 0.036749, 0.073454, 0.073398]],

                           [[0.126475, 0.073499, 0.      , 0.      , 0.      ],
                            [0.052891, 0.074026, 0.073197, 0.      , 0.      ],
                            [0.016108, 0.036654, 0.073476, 0.073441, 0.      ],
                            [0.003803, 0.012194, 0.036763, 0.073627, 0.073582],
                            [0.003807, 0.012364, 0.036585, 0.073612, 0.073896]],

                           [[0.052861, 0.073511, 0.074239, 0.      , 0.      ],
                            [0.015992, 0.036578, 0.073537, 0.074014, 0.      ],
                            [0.003824, 0.012296, 0.037022, 0.073046, 0.073915],
                            [0.003723, 0.01218 , 0.036664, 0.073279, 0.073467],
                            [0.003719, 0.012279, 0.036952, 0.073343, 0.073559]],

                           [[0.016125, 0.03695 , 0.073882, 0.073944, 0.      ],
                            [0.003835, 0.012325, 0.036688, 0.0736  , 0.073073],
                            [0.003905, 0.012229, 0.037068, 0.073494, 0.073918],
                            [0.003924, 0.012213, 0.036678, 0.073292, 0.073739],
                            [0.003738, 0.012222, 0.036522, 0.073415, 0.073221]],

                           [[0.003872, 0.012061, 0.036613, 0.073295, 0.073823],
                            [0.003762, 0.012115, 0.03685 , 0.073617, 0.07348 ],
                            [0.003811, 0.012509, 0.036613, 0.073714, 0.074037],
                            [0.003729, 0.01234 , 0.036697, 0.073529, 0.073794],
                            [0.003835, 0.012573, 0.036643, 0.073608, 0.07308 ]]])

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0)
        return r

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        obs = self.state
        demand = self.demand()
        obs2 = self.transition(obs, action, demand)
        self.state = obs2
        reward = self.reward(obs, action, obs2)
        done = 0
        return obs2, reward, done, {}

    def _reset(self):
        return self.state

