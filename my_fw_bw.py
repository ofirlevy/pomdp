import random
import numpy
import matplotlib.pyplot as plt
import pickle
from data_utils import *
import pandas as pd

np.set_printoptions(suppress=True, linewidth=500)
pd.set_option('display.width', 1000)

# computing forward and backward probabilities
# forward:
# alpha_t(j) = P(o1, ..., ot, qt = j | HMM)
# forward function: returns numpy matrix of size (N, T)
def forwardprobs(observations, initialprob, trans, emis, numstates, action):
    forwardmatrix = numpy.zeros((numstates, len(observations)))
       
    # initialization    
    for s in range(numstates):
        forwardmatrix[ s, 0 ] = initialprob[s] * emis[action[0], s, observations[0]]
    
    # recursion step
    for t in range(1, len(observations)-1):
        for s in range(numstates):            
            forwardmatrix[s, t] = emis[action[t+1], s, observations[t+1]] * sum([forwardmatrix[s2, t-1] * trans[action[t], s, s2] \
                                       for s2 in range(numstates)])
    # handle the last step
    t = len(observations)-1
    for s in range(numstates):
        forwardmatrix[s, t] = sum([forwardmatrix[s2, t-1] * trans[action[t], s, s2] \
                                   for s2 in range(numstates)])    
    
    return forwardmatrix

# beta_t(j) = P(o_{t+1}, ..., o_T | qt = j, HMM)
# backward function: returns numpy matrix of size (N, T)
def backwardprobs(observations, trans, emis, numstates, action):
    backwardmatrix = numpy.zeros((numstates, len(observations)))

    # initialization
    for s in range(numstates):
        backwardmatrix[ s, len(observations) - 1 ] = 1.0
        #backwardmatrix[ s, len(observations) - 2 ] = 1.0

    # recursion
    for t in range(len(observations) - 2, -1, -1):
        for s in range(numstates):
            backwardmatrix[s, t] = sum([ trans[action[t], s2, s] * emis[action[t+1], s2, observations[t+1]] * backwardmatrix[s2, t+1] \
                                         for s2 in range(numstates) ])         
            
    return backwardmatrix                               
   
####
# expectation step:
# re-estimate xi_t(i, j) and gamma_t(j)
# returns two things:
# - gamma is a (N, T) numpy matrix
# - xi is a list of T numpy matrices of size (N, N)
def expectation(observations, trans, emis, numstates, forward, backward, action):
    # denominator: P(O | HMM)
    p_o_given_hmm = sum([forward[s_i, len(observations) -1] for s_i in range(numstates) ])
    
    # computing xi
    xi = [ ]
    for t in range(len(observations) - 2):
       
        xi_t = numpy.zeros((numstates, numstates))
        
        for s_i in range(numstates):
            for s_j in range(numstates):
                xi_t[ s_i, s_j] = (forward[s_i, t] * trans[action[t+1], s_j, s_i] * emis[action[t+2], s_j, observations[t+2]] * backward[s_j, t+2]) / p_o_given_hmm
        xi.append(xi_t)

    # computing gamma
    gamma = numpy.zeros((numstates + 2, len(observations)-1))
    for t in range(len(observations) - 2):
        for s_i in range(numstates):
            gamma[s_i, t] = sum([ xi[t][s_i, s_j] for s_j in range(numstates) ])

    for s_j in range(numstates):
        gamma[s_j, len(observations) - 2] = sum( [ xi[t][s_i, s_j] for s_i in range(numstates) ] )
            
    return (gamma, xi)

###
# maximization step:
# re-estimate trans, emis based on gamma, xi
# returns:
# - initialprob
# - trans
# - emis
def maximization(observations, gamma, xi, numstates, vocabsize, action):
    # re-estimate initial probabilities
    initialprob = numpy.array([gamma[s_i, 0] for s_i in range(numstates)])
    
    # re-estimate emission probabilities
    emis = numpy.zeros((numstates, vocabsize))

    for s in range(numstates):
        denominator = sum( [gamma[s, t] for t in range(len(observations)-1)])
        if (denominator == 0): denominator = 1
        for vocab_item in range(vocabsize):
            emis[s, vocab_item] = sum( [gamma[s, t] for t in range(len(observations)-1) if observations[t] == vocab_item] )/denominator

    # re-estimate transition probabilities
    trans = numpy.zeros((numstates, numstates))

    for s_i in range(numstates):
        # xi[t] corresponds to action[t+1]
        denominator = 0
        for t in range(len(observations) - 1): 
            if (action[t+1] == 0):
                denominator += gamma[s_i, t] 
        
        if (denominator == 0): denominator = 1
        
        for s_j in range(numstates):
            for t in range(len(observations) - 2): 
                if (action[t+1] == 0):
                    trans[s_j, s_i] += xi[t][s_i, s_j]                    
            trans[s_j, s_i] = trans[s_j, s_i] / denominator


    return (initialprob, trans, emis)

##########
# testing forward/backward
def run_forwardbackward(numiter,num_of_seqs):
    
    # capacity
    c = 10
    # number of states
    numstates = c + 2
    vocabsize = c + 1     ## all possible outputs (rewards)
    
    
    T = 200
    p_rest = np.random.uniform(0.4,0.8, num_of_seqs)

    #real_p = np.array([0.6,0.9,0.6,0.9,0.0]) # the last element is meaningless
    #real_p = np.array([0.9,0.6,0.9,0.6,0.0]) # the last element is meaningless
    
    #p = np.array([0.55,0.55,0.55,0.55,0.0]) # the last element is meaningless
    #p = np.array([0.6,0.9,0.6,0.9,0.0]) # the last element is meaningless
    
    real_p = np.array([0.99,0.99,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.0]) # the last element is meaningless
    p = np.random.random(real_p.shape)
    
    
    # generate award/reward sequence
    (action, reward, state) = generate_seq(num_of_seqs,c,real_p,T,p_rest)
    
    
    #fileObject = open("data_small",'r')
    #seq = pickle.load(fileObject)
    #(action, reward, state) = seq
   
  
    # HMM initialization
    
    # initialize initial probs
    # At t=0 we are at the highest state        
    initialprob = numpy.zeros(numstates)    
    initialprob[numstates-1] = 1.0    
    
    # initialize emission probs
    emis = get_emis_mat(numstates, vocabsize)      
    
    
    # initialize transition probs
    trans = get_trans_mat(c, p)
    ## trans[to,from]

    calc_trans = np.zeros((num_of_seqs, trans.shape[1], trans.shape[2]))
    
    for iteration in range(numiter):        
        for seq_num in range(num_of_seqs):
    
            forward = forwardprobs(reward[seq_num], initialprob, trans, emis, numstates, action[seq_num])
            backward = backwardprobs(reward[seq_num], trans, emis, numstates, action[seq_num])
    
            gamma, xi = expectation(reward[seq_num], trans, emis, numstates, forward, backward, action[seq_num])
    
            initialprob, calc_trans[seq_num], bla = maximization(reward[seq_num], gamma, xi, numstates, vocabsize, action[seq_num])
            
        #print("EMIS")
        #print(emis)
        #print("\n")
        
        trans[0] = np.mean(calc_trans,axis=0)
        
    # return the estimated probability    
    p_debug = np.zeros(trans.shape[0])
    for i in range(trans.shape[0]):
        p_debug[i] = trans[0,i+1,i]
                
    return real_p, p_debug
            


def main():
    
    # testing sequence 5 trys for each number of sequence
    num_of_seqs = np.array([2,5,10,25,50,100,150,200,300])
    numiter = 30
    out_diff = np.zeros((num_of_seqs.shape[0],5))
    
    for seq_num in range(num_of_seqs.shape[0]):
        for take_num in range(5):
        # display some lines
            real_p, p_est = run_forwardbackward(numiter, num_of_seqs[seq_num])
            print('number of sequences:')
            print(num_of_seqs[seq_num])
            print(p_est)
            print(np.mean(abs(p_est-real_p[:-1])))
            out_diff[seq_num,take_num] = np.mean(abs(p_est-real_p[:-1]))
    
    fileObject = open("out",'wb') 
    pickle.dump(out_diff,fileObject)   
    fileObject.close()
    
    
    
if __name__ == "__main__": main()

