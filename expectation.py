import numpy as np
def estimate_prob(action, state, N):
    cnt_all = np.zeros(N)
    cnt_up = np.zeros(N)    
    for i in xrange(0, action.shape[0]):
        for t in xrange(1, action.shape[1]):
            if (action[i,t] == 0):
                cnt_all[state[i,t-1]] += 1
                if (state[i,t] == state[i,t-1] + 1):
                    cnt_up[state[i,t-1]] += 1
           
    prob = cnt_up/cnt_all
    return prob
    
    
# doing a similar job in verctorize way:
#prev_state = np.roll(state,1)    
#xx = state[state == prev_state + 1]
#cnt_up2 = np.bincount(xx)
#yy = prev_state[action==0]
#yy[yy==-1]=0
#cnt_all2 = np.bincount(yy)
