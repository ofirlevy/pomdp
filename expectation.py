import numpy as np
def estimate_prob(action, state, N):
    # p(state) = action = 0 and state[t] = state[t-1] + 1
    cnt_all = np.zeros(N)
    cnt_up = np.zeros(N)
    #state = state + 1  # to match indexes
    for i in xrange(1, action.shape[0]):
        for t in xrange(1, action.shape[1]):
            if (action[i,t] == 0):
                cnt_all[state[i,t-1]] += 1
                if (state[i,t] == state[i,t-1] + 1):
                    cnt_up[state[i,t-1]] += 1
                #assert np.abs(state[i,t] - state[i,t-1]) <= 1  # sanity check
    # TODO - replace with np.count_nonzero(np.logical_and(state[i]==3 , action[i]==0))                    
    prob = cnt_up/cnt_all
    #prob[cnt_all == 0] = 1 # np.random.random()
    #prob[prob < 0.1] = 0.1
    return prob

    
    
    