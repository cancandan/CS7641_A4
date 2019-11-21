from collections import defaultdict
import numpy as np
from gym.envs.toy_text import discrete
import random
class Simple(discrete.DiscreteEnv):
    def __init__(self):
        t = {
            'leisure': {
                'facebook': {'leisure': 0.9, 'class1': 0.1},
                'quit': {'leisure': 0.1, 'class1': 0.9},
                'study': {},
                'sleep': {},
                'pub': {}
            },
            'class1': {
                'study': {'class2': 0.6, 'leisure': 0.4},
                'facebook': {'class2': 0.4, 'leisure': 0.6},
                'quit': {},
                'sleep': {},
                'pub': {}
            },
            'class2': {
                'study': {'class3': 0.5, 'end': 0.5},
                'sleep': {'end': 0.5, 'class3': 0.5},
                'facebook': {},
                'quit': {},
                'pub': {},
            },
            'class3': {
                'study': {'end': 0.6, 'class1': 0.08, 'class2': 0.16, 'class3': 0.16},
                'pub': {'end': 0.4, 'class1': 0.12, 'class2': 0.24, 'class3': 0.24},
                'facebook': {},
                'quit': {},
                'sleep': {}
            },
            'end': {}
        }
        rewards = {
            'class1': 4,
            'class2': 6,
            'class3': 10,
            'leisure': -1,
            'end': 0
        }
        terminals = ['end']

        init = 'class1'

        states = list(t.keys())

        actions = []
        for state in t.keys():
            actions.extend(t[state])
        actions = list(set(actions))

        self.nA = len(actions)
        self.nS = len(states)

        states = list(t.keys())
        actions = list(t[states[0]].keys())

        states = dict(zip(states, range(len(states))))
        actions = dict(zip(actions, range(len(actions))))

        self.s = 1
        self.states=states
        self.actions=actions
        P = defaultdict(lambda: defaultdict(dict))
        for src, srcidx in states.items():
            for act, actidx in actions.items():
                if t.get(src):
                    if t[src].get(act):
                        li = []
                        P[srcidx][actidx] = li
                        for trg, trgidx in states.items():
                            if t[src][act].get(trg):
                                li.append((t[src][act][trg], trgidx, rewards[trg], True if trg == 'end' else False))

        # super(Simple, self).__init__(self.nS, self.nA, P, np.ones(5))

        class Myacs(object):
            def sample(self):
                return random.choice(list(self.P[self.s].keys()))
            def seed(self,p):
                pass


        self.action_space = Myacs()
    # def step(self, a):
    #     transitions = self.P[self.s][a]
    #     i = categorical_sample([t[0] for t in transitions], self.np_random)
    #     p, s, r, d= transitions[i]
    #     self.s = s
    #     self.lastaction = a
    #     return (s, r, d, {"prob" : p})