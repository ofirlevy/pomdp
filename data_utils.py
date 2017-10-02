import numpy as np
import matplotlib.pyplot as plt

# create Transition matrix for each action
def get_trans_mat(c, p):
    # colomn sf state that you were, raws are state that you end up given the action
    trans = np.zeros([c+1,c+2,c+2])
    
    # initialize the T0 matrix using input probablity P 
    for i in xrange(0, c+1):
        trans[0,i,i] = 1-p[i]
        trans[0,i+1,i] = p[i]
    trans[0,c+1,c+1] = 1
    
    # Ta is a (c+2)x(c+2) matrix 
    for a in xrange(1, c+1):
        trans[a,0,0:a+1] = 1
        for i in xrange(1, c-a+2):
            trans[a,i,i+a] = 1
    
    return trans


# TODO make a class
# generate action/reward/state sequence 
def generate_single_seq(c,p,T,p_rest):
    # create single state, action, and award sequences in the length of T
    state = np.empty(T)
    state[:] = np.NAN
    action = np.zeros(T)
    reward = np.zeros(T)
    
    # state 0 means F, 1 means 0 etc, 11 means 10
    
    # TODO consider switch to represt single state/action as vector
    state[0] = c
    
    for t in xrange(1,T):  
        # give more probability to rest regardless of the state
        if np.random.random() < p_rest:
            # rest time
            # go one state up with probablility according to the current state
            action[t] = 0
            state[t] = state[t-1]
            if np.random.random() < p[int(state[t-1]+1)]:
                state[t] = min(state[t]+1, c)        
        else:
            # action time
            # draw action value, normal dist around previous state
            mu, sigma = 0, 3
            #action[t] = int(np.random.normal(mu, sigma, 1) + (state[t-1]-1))
            action[t] = np.random.randint(11, size=1)
            action[t] = min(action[t], c)
            action[t] = max(action[t], 0)
        
            # calculate the reward
            reward[t] = max(min(state[t-1],action[t]),0)
            # calculate the new state    
            state[t] = max(-1, state[t-1]-action[t])      
    
    return (action.astype(int), reward.astype(int), state.astype(int))


def generate_seq(seq_num, c,p,T,p_rest):
    state_seq = np.empty([seq_num,T], dtype=np.int)
    action_seq = np.empty([seq_num,T], dtype=np.int)
    reward_seq = np.empty([seq_num,T], dtype=np.int)
    
    for i in xrange(0,seq_num):
        (x, y, z) = generate_single_seq(c,p,T,p_rest)
        action_seq[i] = x
        reward_seq[i] = y
        state_seq[i] = z
        
    return (action_seq, reward_seq, state_seq)








