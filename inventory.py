import numpy as np

class MDP(object):
    def __init__(self):
        self.gamma = 0.9
        pass

    def reset(self):
        raise NotImplementedError

    # return r, sp after taking action a from state s
    def step(self,s,a):
        sp_list, sp_dist = self.transition_func(s,a)
        sp = int(np.random.choice(sp_list, 1, p=sp_dist))
        r = self.reward_func(s,a,sp)
        return r, sp

    # return a tuple of (next_states, probabilities)
    def transition_func(self, s, a):
        raise NotImplementedError

    # return the reward r(s,a)
    def reward_func(self, s, a, sp):
        raise NotImplementedError

    # return whether or not the current state is a terminal state
    def done(self, s):
        raise NotImplementedError

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return []

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        raise NotImplementedError

    def render(self, s):
        pass

class Inventory(MDP):
    def __init__(self, param=0):
        self.K = 4 # fixed cost of ordering
        self.c = 2 # variable cost of ordering
        self.h = lambda n: max(n, -3*n) # cost of holding n units
        self.f = lambda n: 8*n # revenue from selling n units
        self.demands = np.arange(5)
        # self.demand_models = [[0.25, 0.5, 0.25, 0., 0.],
        #                       [0.1, 0.1, 0.5, 0.3, 0.],
        #                       [0, 0.1, 0.3, 0.3, 0.3],
        #                       [0, 0.1, 0.2, 0.2, 0.5]]
        self.demand_models = np.array( [[0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.]] )
        self.param = param
        self.gamma = 0.7
        self.M = 5

        self.nA = 5
        self.nS = 10
        self.P = self.trn()
    # reset environment and return s0
    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        sp = [min(s+a-d,self.M) for d in self.demands]
        return (sp, self.demand_models[self.param])

    def trn(self):
        P = {s: {a: [] for a in range(self.M)} for s in range(self.M)}
        for s in range(self.M):
            for a in range(self.M):
                li = P[s][a]
                sp, probs = self.transition_func(s,a)
                for k in range(len(sp)):
                    rew = self.reward_func(s,a,sp[k])
                    li.append((probs[k],sp[k],rew,False))
        return P


    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        order_cost = 0
        if a > 0:
            order_cost = self.K + self.c*a;
        holding_cost = self.h(s+a)
        revenue = self.f(min(s+a - sp, s+a))
        return revenue - holding_cost - order_cost

    # return if s is terminal
    def done(self, s):
        return False

    # return a list of all the actions in the MDP
    def action_space(self,s):
        return np.arange(self.M-s)

    def render(self, s, a=0):
        print("Inventory:", s, "\tAction:", a)



class TreatmentPlan(MDP):
    def __init__(self, param=0):
        self.param = param
        self.allergies = np.array( [[1,0,0],
                                    [0,1,0],
                                    [0,0,1],
                                    [0,0,0]] )
        self.drug_transitions = np.array([ [[0.0, 0.4, 0.6],
                                            [0.0, 0.5, 0.5],
                                            [0.0, 0.8, 0.2]],
                                           [[0.5, 0.5, 0.0],
                                            [0.4, 0.2, 0.4],
                                            [0.2, 0.8, 0.0]] ])
        self.gamma = 1.0

    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        sp_list = np.maximum( np.minimum(s + np.arange(-1, 1), 4), 0 )
        sp_dist = self.drug_transitions[self.allergies[self.param, a], a]
        return sp_list, sp_dist

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        rewards = [-1.0, -0.2, -0.2, 0.5]
        #rewards =  [0, 2.0, 1.5, 1.0, 0.9, 0.5, 0.0]
        return rewards[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return False

    # return a list of all the states of the MDP
    def state_space(self):
        return np.arange(7)

    # return a list of all the actions in the MDP
    def action_space(self,s):
        if s == 0:
            return np.arange(4)
        else:
            return [0]