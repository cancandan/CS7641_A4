import numpy as np
import time
import random
import random as rand
RS = 903280523


class Gambler(object):
    def __init__(self):
        self.goal=10
        self.terminal=0
        self.startMoney=5
        self.phead=0.4
        self.currentMoney=self.startMoney
    def transitions(self, s, a):
        if s==self.terminal or s==self.goal:
            return [(1,s,0,True)]
        else:
            if a>s:
                raise ("can not act with more money than you have")
            if a<=0:
                raise ("money played should be positive")

            loose = max(s-a,self.terminal)
            win = min(s+a,self.goal)

            return [(1-self.phead, loose, -a, True if loose==0 else False),
                    (self.phead, win, a, True if win==self.goal else False)]
    # s can be 1 to 99
    def availableActions(self,s):
        if s==self.terminal or s==self.goal:
            return [0]
        actions = list(range(1, s + 1))
        return actions
    def allPossibleStates(self):
        return list(range(self.goal+1))

    def allPossibleActions(self):
        return list(range(1,self.goal))

    def sampleAction(self,s):
        return random.choice(self.availableActions(s))

    def step(self,a):
        chance = random.uniform(0, 1)
        # new_state, reward, done, info
        if chance < self.phead:
            win = min(self.currentMoney + a, self.goal)
            self.currentMoney = win
            return (win, a, True if win == self.goal else False)
        else:
            loose = max(self.currentMoney - a, self.terminal)
            self.currentMoney = loose
            return (loose, -a, True if loose == 0 else False)

    def reset(self):
        self.currentMoney=self.startMoney

def setMySeed(seed=RS):
    rand.seed(seed)
    np.random.seed(seed)

def sbvaliter(gambler, gamma, theta):
    V = {st:0 for st in gambler.allPossibleStates()}
    mainLoopCounter = 0
    deltas=[]
    begin = time.time()
    while True:
        delta = 0
        for s in V:
            v = V[s]
            themax = float('-inf')
            for ac in gambler.availableActions(s):
                thesum = 0
                for ns in gambler.transitions(s, ac):
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                if thesum > themax:
                    themax = thesum
            V[s] = themax
            delta = max(delta, np.abs(v - V[s]))
        mainLoopCounter += 1
        deltas.append(delta)
        if delta < theta:
            break
    pi = {st:0 for st in gambler.allPossibleStates()}
    for s in pi:
        themax = float('-inf')
        action_vals = {}
        for ac in gambler.availableActions(s):
            thesum = 0
            for ns in gambler.transitions(s, ac):
                thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
            action_vals[ac] = thesum
        sorted_action_vals = sorted(action_vals.items(), key=lambda y: (y[1], -y[0]))
        if sorted_action_vals:
            pi[s] = sorted_action_vals[-1][0]
    duration = time.time() - begin
    print("Total iterations:{} Duration:{}".format(mainLoopCounter, duration))
    print (pi.values())
    return pi, V, deltas, mainLoopCounter, duration


def sbpoliter(gambler, gamma, theta):
    V = {st: 0 for st in gambler.allPossibleStates()}
    pi = {st: gambler.availableActions(st)[0] for st in gambler.allPossibleStates()}
    mainLoopCounter = 0
    deltas = []
    begin = time.time()
    while True:
        while True:
            delta = 0
            for s in V:
                v = V[s]
                thesum = 0
                for ns in gambler.transitions(s,pi[s]):
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                V[s] = thesum
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break
        polstable = True
        for s in V:
            olda = pi[s]
            action_vals = {}
            for na in gambler.availableActions(s):
                thesum = 0
                for ns in gambler.transitions(s,na):
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                action_vals[na] = thesum
            sorted_action_vals = sorted(action_vals.items(), key=lambda y: (y[1], -y[0]))
            pi[s] = sorted_action_vals[-1][0]
            if olda != pi[s]:
                polstable = False
        deltas.append(delta)
        mainLoopCounter += 1
        if polstable:
            print (pi.values())
            duration = time.time() - begin
            print("Total iterations:{} Duration:{}".format(mainLoopCounter, duration))
            return pi, V, deltas, polstable, mainLoopCounter,duration


from collections import defaultdict
import operator
def qlearning(gambler, total_episodes=100000, theta=0.1, alpha=0.1, epsilon=0.1, gamma=0.95, epsilon_decay=0.99, alpha_decay=0.99, max_steps=1000):
    setMySeed(RS)
    states = gambler.allPossibleStates()
    q = defaultdict(float)

    for s in states:
        for a in gambler.availableActions(s):
            q[s,a]=0

    min_epsilon = 0.1
    min_alpha = 0.1
    rewards = []
    times = []
    episodeLengths = []
    maxDeltas=[]
    for episode in range(total_episodes):
        maxDelta = 0
        gambler.reset()
        state = gambler.startMoney
        episode_rewards = 0
        begin = time.clock()
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)
            availableActions = gambler.availableActions(state)
            if exp_exp_tradeoff > epsilon:
                action = max(availableActions, key=lambda a1: q[state, a1])
            else:
                action = gambler.sampleAction(state)
            new_state, reward, done = gambler.step(action)
            maxactionval = max(q[new_state, a] for a in gambler.availableActions(new_state))
            td_delta = reward + gamma * maxactionval - q[state, action]
            q[state, action] = q[state, action] + alpha * td_delta
            maxDelta = max(maxDelta, abs(td_delta))
            episode_rewards = episode_rewards + reward

            state = new_state

            if done:
                break
        duration = time.clock() - begin

        maxDeltas.append(maxDelta)
        episodeLengths.append(step)
        rewards.append(episode_rewards)
        times.append(duration)

        alpha *= alpha_decay
        if alpha < min_alpha:
            alpha = min_alpha

        epsilon *= epsilon_decay
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        if episode > 10000 and episode % 10000 == 0:
            print("Episode:{}".format(episode))

    return q, maxDeltas, rewards, episodeLengths, times


if __name__ == '__main__':
    gambler = Gambler()
    sbvaliter(gambler, 0.9, 1e-3)
    # sbpoliter(env, 0.9, 1e-3)
    # q, maxDeltas, rewards, episodeLengths, times = qlearning(gambler,500000, alpha=0.9, epsilon=1, gamma=0.9, epsilon_decay=0.99, alpha_decay=0.99, max_steps=1000)
    # pi = []
    # for s in gambler.allPossibleStates():
    #     best = max(gambler.availableActions(s), key=lambda a1: q[s, a1])
    #     pi.append(best)
    # print (pi)