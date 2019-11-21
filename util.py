import random as rand
import numpy as np
from IPython.display import clear_output
from time import sleep
import joblib
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from collections import namedtuple
import itertools
import pandas as pd
import sys
import random

from gym.envs.toy_text.frozen_lake import generate_random_map
import gym

RS = 903280523

def setMySeed(seed=RS):
    rand.seed(seed)
    np.random.seed(seed)

def deleteEnvironment(env_name):
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

def jack():
    env_name='Jack-v0'
    deleteEnvironment(env_name)
    gym.envs.register(
        id=env_name,
        entry_point='gymEnvs:JacksCarRental'
    )
    return gym.make(env_name)

def hanoi():
    env_name = 'Hanoi-v0'
    deleteEnvironment(env_name)
    gym.envs.register(
        id=env_name,
        entry_point='gymEnvs:HanoiEnv'
    )
    return gym.make(env_name)

def simple():
    env_name = 'Simple-v0'
    deleteEnvironment(env_name)
    gym.envs.register(
        id=env_name,
        entry_point='gymEnvs:Simple'
    )
    return gym.make(env_name)

def guess():
    env_name = 'Guess-v0'
    deleteEnvironment(env_name)
    gym.envs.register(
        id=env_name,
        entry_point='gymEnvs:GuessingGame'
    )
    return gym.make(env_name)

def myInventory():
    env_name="Inventory-v0"
    deleteEnvironment(env_name)
    gym.envs.register(
        id='Inventory-v0',
        entry_point='gymEnvs:InventoryEnv',
    )
    return gym.make(env_name)

def myFrozenLake(size=8,randomMap=True,slippery=False, rewarding=False, equiProbable=True, frozenProb=0.9, seed=RS):
    setMySeed(seed)
    if randomMap:
        map = generate_random_map(size,frozenProb)
        env_name = "MyFrozenLakeMap_size_{}_seed_{}-v0".format(size, seed)
        deleteEnvironment(env_name)
        joblib.dump(map, env_name)
        gym.envs.register(id=env_name,
                          entry_point='gymEnvs:MyFrozenLakeEnv',
                          kwargs={'desc': map, 'is_slippery': slippery, 'rewarding': rewarding,
                                  'equiProbable': equiProbable},
                          max_episode_steps=size ** 4, reward_threshold=0.78
                          )
        return gym.make(env_name)
    else:
        env_name = "MyFrozenLakeMapCustom-v0".format(size, seed)
        deleteEnvironment(env_name)
        gym.envs.register(id=env_name,
                          entry_point='gymEnvs:MyFrozenLakeEnv',
                          kwargs={'map_name': '20x20', 'is_slippery': slippery, 'rewarding': rewarding,
                                  'equiProbable': equiProbable},
                          max_episode_steps=1000, reward_threshold=0.78
                          )
        return gym.make(env_name)


def sbpoliter(env, gamma, theta):
    V = np.zeros(env.nS)
    pi = np.zeros(env.nS, dtype=np.int8)
    mainLoopCounter = 0
    deltas = []
    begin = time.time()
    while True:
        while True:
            delta = 0
            for s in range(env.nS):
                v = V[s]
                thesum = 0
                for x in env.P[s][pi[s]]:
                    thesum = thesum + x[0] * (x[2] + gamma * V[x[1]])
                V[s] = thesum
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break
        polstable = True
        for s in range(env.nS):
            olda = pi[s]
            action_vals = {}
            for na in env.P[s]:
                thesum = 0
                for ns in env.P[s][na]:
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                action_vals[na] = thesum
            sorted_action_vals = sorted(action_vals.items(), key=lambda y: (y[1], -y[0]))
            pi[s] = sorted_action_vals[-1][0]
            if olda != pi[s]:
                polstable = False
        deltas.append(delta)
        mainLoopCounter += 1
        if polstable:
            duration = time.time() - begin
            print("Total iterations:{} Duration:{}".format(mainLoopCounter, duration))
            return pi, V, deltas, polstable, mainLoopCounter,duration


def sbvaliter(env, gamma, theta):
    V = np.zeros(env.nS)

    mainLoopCounter = 0
    deltas=[]
    begin = time.time()
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            themax = float('-inf')
            for na in env.P[s]:
                thesum = 0
                for ns in env.P[s][na]:
                    thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
                if thesum > themax:
                    themax = thesum
            V[s] = themax
            delta = max(delta, np.abs(v - V[s]))
        mainLoopCounter += 1
        deltas.append(delta)
        if delta < theta:
            break
    pi = np.zeros(env.nS, dtype=np.int8)
    for s in range(env.nS):
        themax = float('-inf')
        action_vals = {}
        for na in env.P[s]:
            thesum = 0
            for ns in env.P[s][na]:
                thesum = thesum + ns[0] * (ns[2] + gamma * V[ns[1]])
            action_vals[na] = thesum
        sorted_action_vals = sorted(action_vals.items(), key=lambda y: (y[1], -y[0]))
        pi[s] = sorted_action_vals[-1][0]
    duration = time.time() - begin
    print("Total iterations:{} Duration:{}".format(mainLoopCounter, duration))
    return pi, V, deltas, mainLoopCounter, duration


from matplotlib.colors import PowerNorm
from matplotlib.colors import SymLogNorm
def plotFLPolicy(pi,v,env,size):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(v.reshape(size, size), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(size) - .5)
    ax.set_yticks(np.arange(size) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:size, 0:size]
    a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
    Pi = np.reshape(np.argmax(pi, axis=1), (size, size))
    for y in range(size):
        for x in range(size):
            a = Pi[y, x]
            u, v = a2uv[a]
            letter = str(env.unwrapped.desc[y, x].item().decode())
            if not (letter=="H" or letter=="G"):
                plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)

            plt.text(x, y, letter,
                     color='r' if letter=="H" else 'g', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    ax.set_ylim(len(Pi) - 0.5, -0.5)
    plt.grid(color='b', lw=2, ls='-')
    fig.tight_layout()


from collections import deque
def qlearning(env, total_episodes=100000, theta=0.1, alpha=0.1, epsilon=0.1, gamma=0.95, epsilon_decay=0.99, alpha_decay=0.99, max_steps=1000):
    setMySeed(RS)
    env.action_space.seed(RS)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    min_epsilon = 0.1
    min_alpha = 0.1
    qtable = np.ones((state_size, action_size))*0
    rewards = []
    times = []
    episodeLengths = []

    maxDeltas=[]
    hitgoal = 0
    firstHitGoal = None
    for episode in range(total_episodes):
        maxDelta = 0
        state = env.reset()
        episode_rewards = 0
        begin = time.clock()
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            if reward==10:
                if firstHitGoal is None:
                    firstHitGoal = episode
                    print (firstHitGoal)
                hitgoal+=1
            td_delta = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
            qtable[state, action] = qtable[state, action] + alpha * td_delta
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
        if episode > 5000 and episode % 5000 == 0:
            print("Episode:{} NumHitGoal:{}".format(episode, hitgoal))

    return qtable, maxDeltas, rewards, episodeLengths, times

def policyScore(env, pi, q=False):
    setMySeed(RS)
    env.action_space.seed(RS)
    total=0
    numEpisodes = 300
    totalHoles = 0
    for i in range(numEpisodes):
        s = env.reset()
        episodeReward = 0
        while True:
            if q:
                a = np.argmax(pi[s])
            else:
                a = pi[s]
            s, reward, done, info = env.step(a)
            episodeReward = episodeReward + reward
            if done:
                if reward==-10:
                    totalHoles+=1
                break
        total = total + episodeReward
    return total / numEpisodes, totalHoles


def flpoldiff(pol1,pol2,flsize=20):
    diff = np.where(pol1.reshape(flsize, flsize) != pol2.reshape(flsize, flsize))
    return list(zip(diff[0], diff[1]))


def fl_policy_viz(pi, v, env, size, yellows):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(v.reshape(size, size), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(size) - .5)
    ax.set_yticks(np.arange(size) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:size, 0:size]
    a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
    Pi = np.reshape(np.argmax(pi, axis=1), (size, size))
    for y in range(size):
        for x in range(size):
            a = Pi[y, x]
            u, v = a2uv[a]
            letter = str(env.unwrapped.desc[y, x].item().decode())
            if not (letter == "H" or letter == "G"):
                plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)

            if letter != 'H':
                thecolor = 'g'
            if (y, x) in yellows:
                thecolor = 'y'
            if letter == 'H':
                thecolor = 'r'

            plt.text(x, y, letter,
                     color=thecolor, size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    ax.set_ylim(len(Pi) - 0.5, -0.5)
    plt.grid(color='b', lw=2, ls='-')
    fig.tight_layout()


if __name__ == '__main__':
    # env = simple()
    # sbvaliter(env,0.9,1e-3)
    # qtable, maxDeltas, rewards, episodeLengths, times = qlearning(env,1000, alpha=0.1, epsilon=1, gamma=0.99, epsilon_decay=.9999, alpha_decay=1, max_steps=1000)
    env = myInventory()
    sbvaliter(env, 0.9, 1e-3)