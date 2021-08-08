import gym
import numpy as np

#use epsilon greedy, 1-epsilon select action with max Q and epsilon select other actions
def eps_greedy(Q, s, eps=0.1):
    if np.random.uniform(0, 1) < eps:
        return np.random.randint(Q.shape[1])
    else:
        return greedy(Q, s)

#Select action with max cummulative reward
def greedy(Q, s):
    return np.argmax(Q[s])

#for test
def run_episodes(env, Q, num_episodes=100, to_print=False):
    tot_rew = []
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # select a greedy action
            next_state, rt, done, _ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rt
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))

    return np.mean(tot_rew)

#main Function
if __name__ == '_main_':
    #use network environment
    env = gym.make('networkenv')
    #number of task 
    M=25
    #number of MD
    N=5
    #bandwidth for both uplink and downlink between a user and an edge server is 150 MB
    BW=150
    #rate of the CPU of an edge sever is 9*10^8 cycle/s
    CPU=900000000
    #learning rate
    lr = 0.01
    #number of episodes
    num_episodes = 5000
    eps = 0.4
    gamma = 0.95
    eps_decay = 0.001

    nA = env.action_space.n
    nS = env.observation_space.n

    Q = np.zeros((nS, nA))
    offload_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        if eps > 0.01:
            eps -= eps_decay

        action = eps_greedy(Q, state, eps)

        while not done:
            next_state, rt, done, _ = env.step(action)
            #update Q(S,A) in SARSA
            next_action = eps_greedy(Q, next_state, eps)
            Q[state][action] = Q[state][action] + lr * (rt + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action
            tot_rew += rt
            '''
            reward if state(t)>state(t+1) reward= -1, if state(t)=state(t+1) rewared=0,\
            if state(t)<state(t+1) reward= 1
            '''
            if state < next_state:
                rt = -1
            elif state > next_state:
                rt = 1
            else:
                rt = 0
            if done:
                offload_reward.append(tot_rew)

        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    print(Q)
