import matplotlib.pyplot as plt
import numpy as np
import my_viterbi
from expectation import *
from data_utils import *
import numpy as np
import matplotlib.pyplot as plt



np.set_printoptions(suppress=True, linewidth=240)
np.set_printoptions(formatter={'float': '{: 0.3g}'.format})


# definitions
# # # # # # # 
# capacity
c = 10
# number of states
N = c + 2
# probability to go one state up for a single rest action
# p = [Pf, P0, P1, ....Pc-1]
#real_p = np.array([0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.99,0.0]) # the last element is meaningless
#real_p = np.array([0.99,0.99,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.0]) # the last element is meaningless
real_p = np.array([0.2,0.3,0.4,0.5,0.6,0.6,0.7,0.8,0.85,0.9,0.95,0.0])

p = np.array([0.99,0.22,0.22,0.22,0.22,0.22,0.22,0.22,0.22,0.22,0.22,0.0])

seq_num = 100  # number of sequences
T = 2000  # length of each sequence

p_rest = np.random.uniform(0.55,0.65, seq_num)

(action, reward, state) = generate_seq(seq_num,c,real_p,T,p_rest)

# # # 

# need to count the interval that pass 

#for each pair of reward 

# I know that I'm in failure mode if reward < action.
# if no action was in between 

# go for all action > 0 pairs
# for each pair if on the left - reward < action (we are in failure), and on the right reward = 0. then tleft - tright - 1 is the interval 
# that the state left < 1

seq_num, time = np.where(action > 0)
zero_state_cnt = 0
one_state_cnt = 0
state_cnt = np.zeros(real_p.shape) + 5

for st in xrange (1,11):
    zero_state_cnt = 0
    one_state_cnt = 0    
    for i in xrange (seq_num.shape[0]-1):
        left = seq_num[i],time[i]
        right = seq_num[i+1],time[i+1]
        if seq_num[i] != seq_num[i+1]: continue  # comparing pairs from different sequences
        if (reward[left]<10) and (reward[left] == action[left]): continue  # left isn't necessary on failure or zero mode
        if (reward[right] == st-1): zero_state_cnt += 1
        if (reward[right] == st): one_state_cnt += 1
    state_cnt[st] = float(one_state_cnt)/zero_state_cnt

print state_cnt
print np.abs(state_cnt-real_p)

print zero_state_cnt
print one_state_cnt

plt.figure(1)
plt.subplot(211)    
plt.plot(state[0])

plt.subplot(212)
plt.plot(action[0])    
plt.plot(reward[0])        
plt.show()        

print 'done'