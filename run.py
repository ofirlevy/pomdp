import my_viterbi
from expectation import *
from data_utils import *
import numpy as np
import pickle


# definitions
# # # # # # # 
# capacity
c = 10
# number of states
N = c + 2
# probability to go one state up for a single rest action
# p = [Pf, P0, P1, ....Pc-1]
#real_p = np.array([0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.99,0.0]) # the last element is meaningless
real_p = np.array([0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.1,0.0]) # the last element is meaningless
#real_p = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.0]) 
p_rest = 0.8


seq_num = 100  # number of sequences
T = 200  # length of each sequence


# # # 

# states is a [-1, 0, 1, 2, ..., 10]. size is 10
# initialize to zeros execpt last element
#initial_state = np.zeros([c+2,1])
#initial_state[c+1] = 1
#initial_state[7] = 1

# init with trans and initial state
viterbi = my_viterbi.Decoder(N)


# generate award/reward sequence
#(action, reward, state) = generate_seq(seq_num,c,real_p,T,p_rest)

#seq = (action, reward, state)
#fileObject = open("data",'wb') 
#pickle.dump(seq,fileObject)   
#fileObject.close()

fileObject = open("data_rep",'r')
seq = pickle.load(fileObject)
(action, reward, state) = seq


# randomize initial p
#p = np.array([0.8,0.8,0.8,0.8,0.8,0.3,0.3,0.3,0.3,0.3,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.3,0.4,0.5,0.6,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
#p = np.array([0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.8,0.3,0.0])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)
p = np.array([0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])   # np.ones(real_p.shape) * 0.9 # np.random.random(real_p.shape)

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
    print np.mean(p-real_p)
    #print p
    #print new_state[25]-1

print 'done'


