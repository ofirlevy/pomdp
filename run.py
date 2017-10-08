import my_viterbi
from expectation import *
from data_utils import *
import numpy as np
import pickle
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
real_p = np.array([0.99,0.99,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.0]) # the last element is meaningless
#real_p = np.array([0.2,0.3,0.4,0.5,0.6,0.6,0.7,0.8,0.85,0.9,0.95,0.0])




seq_num = 100  # number of sequences
T = 200  # length of each sequence

p_rest = np.random.uniform(0.5,0.99, seq_num)

# # # 



# states is a [-1, 0, 1, 2, ..., 10]. size is 10
# initialize to zeros execpt last element
#initial_state = np.zeros([c+2,1])
#initial_state[c+1] = 1
#initial_state[7] = 1

# init with trans and initial state
viterbi = my_viterbi.Decoder(N)


# generate award/reward sequence
(action, reward, state) = generate_seq(seq_num,c,real_p,T,p_rest)

#seq = (action, reward, state)
#fileObject = open("data",'wb') 
#pickle.dump(seq,fileObject)   
#fileObject.close()

#fileObject = open("data_graph",'r')
#seq = pickle.load(fileObject)
#(action, reward, state) = seq


#est_p = stat_p(action,reward, state)


# randomize initial p
#p = np.array([0.8,0.8,0.8,0.8,0.8,0.3,0.3,0.3,0.3,0.3,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.3,0.4,0.5,0.6,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.7,0.75,0.85])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
p = real_p # np.random.random(real_p.shape)

print ' we start with:'
print p
print ' _______________'

#p = np.ones(real_p.shape) * 0.5

for i in xrange (1000):

    # update transition matrix with the new prob
    trans = get_trans_mat(c, p)    
    
    # provide the action sequence and run viterbi 
    new_state = viterbi.Decode(action,reward, trans)
    
    # run for all action-state pairs to estimate the probability 
    p = estimate_prob(action, new_state, N)

    print i
    print p-real_p
    print np.mean(abs(p-real_p))
    #print p
    #print new_state[25]-1
    
    if (False):
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(new_state[13,50:150]-1)
        plt.plot(state[13,50:150], 'r--')
        
        
        plt.subplot(212)
        plt.plot(action[13,50:150])    
        plt.plot(reward[13,50:150])        
        plt.show()        
        
        
        
        
        
        
        

print 'done'


