import numpy as np
import time
import random
import random as rand
RS = 903280523

class MDP(object):
    def __init__(self):
        self.P = np.array([[[0.5, 0.5], [0.8, 0.2]], [[0, 1], [0.1, 0.9]]])
        self.R = np.array([[5, 10], [-1, 2]])
    def getAllPossibleStates(self):
        return [0,1]
    def getAvailableActions(self):
        return [0,1]
    def getTransitions(self, s, a):
        return [(self.P[a][s][0],0,self.R[a][0],False),(self.P[a][s][1],1,self.R[a][1],False)]



def sbvaliter(gamma, theta):

    mdp = MDP()
    V = {st:0 for st in mdp.allPossibleStates()}
    mainLoopCounter = 0
    deltas=[]
    begin = time.time()
    while True:
        delta = 0
        for s in V:
            v = V[s]
            themax = float('-inf')
            for ac in mdp.availableActions(s):
                thesum = 0
                for ns in mdp.transitions(s, ac):
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                if thesum > themax:
                    themax = thesum
            V[s] = themax
            delta = max(delta, np.abs(v - V[s]))
        mainLoopCounter += 1
        deltas.append(delta)
        if delta < theta:
            break
    pi = {st:0 for st in mdp.allPossibleStates()}
    for s in pi:
        themax = float('-inf')
        action_vals = {}
        for ac in mdp.availableActions(s):
            thesum = 0
            for ns in mdp.transitions(s, ac):
                thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
            action_vals[ac] = thesum
        sorted_action_vals = sorted(action_vals.items(), key=lambda y: (y[1], -y[0]))
        if sorted_action_vals:
            pi[s] = sorted_action_vals[-1][0]
    duration = time.time() - begin
    print("Total iterations:{} Duration:{}".format(mainLoopCounter, duration))
    print (pi.values())
    return pi, V, deltas, mainLoopCounter, duration


def score(pi, initialMoney, times):
    epsteps=[]
    rews=0
    for i in range(times):
        g = Gambler()
        g.currentMoney=initialMoney
        steps = 0
        while True:
            ns, rew, done = g.step(pi[g.currentMoney])
            steps+=1
            if done:
                if rew==1:
                    rews+=1
                break
        epsteps.append(steps)
    return epsteps, rews

def randomPolicy():
    g = Gambler()
    pi=[0]
    for i in range(1,g.goal):
        pi.append(g.sampleAction(i))
    pi.append(0)
    return pi




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

    min_epsilon = 0.01
    min_alpha = 0.01
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
        if episode > 1 and episode % 1000000 == 0:
            print("Episode:{}".format(episode))

    return q, maxDeltas, rewards, episodeLengths, times


if __name__ == '__main__':
    # score(randomPolicy(), 1000000)
    score([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],2)

    # gambler = Gambler()
    # sbvaliter(gambler, 0.9, 1e-3)
    # sbpoliter(env, 0.9, 1e-3)
    # q, maxDeltas, rewards, episodeLengths, times = qlearning(gambler,500000, alpha=0.9, epsilon=1, gamma=0.9, epsilon_decay=0.99, alpha_decay=0.99, max_steps=1000)
    # pi = []
    # for s in gambler.allPossibleStates():
    #     best = max(gambler.availableActions(s), key=lambda a1: q[s, a1])
    #     pi.append(best)
    # print (pi)