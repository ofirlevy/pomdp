
import numpy as np

'''
N: number of hidden states
'''
class Decoder(object):
    def __init__(self, state_num):
        self.N = state_num
        #self.initialProb = initialProb
        #self.transProb = transProb
        #assert self.initialProb.shape == (self.N, 1)
        
    def Decode(self, action, reward, transProb):
        
        trellis = np.zeros((action.shape[0], self.N, action.shape[1]))
        backpt = np.ones((action.shape[0], action.shape[1]), 'int32') * -1
                
        # initialization
        trellis[:,:, 0] = np.zeros(self.N)
        trellis[:,self.N-1, 0] = 1
        
        
        for i in xrange(1, action.shape[0]):
            for t in xrange(1, action.shape[1]):
                
                if (reward[i,t]<action[i,t]):
                    # the current and previous states are well known in this case
                    # update current state to be fail
                    trellis[i,:,t] = np.zeros(self.N)
                    trellis[i,0,t] = 1.0
                    backpt[i, t] = 0 
                    # update previous state according to the reward
                    trellis[i,:, t-1] = np.zeros(self.N)
                    if (reward[i,t] > 0):
                        trellis[i, reward[i,t]+1, t-1] = 1.0
                    else:
                        trellis[i,0, t-1] = 0.5
                        trellis[i,1, t-1] = 0.5
                else:
                    trellis[i,:, t] = transProb[action[i,t]].dot(trellis[i,:, t-1])
                    backpt[i, t] = ( transProb[action[i,t]].dot(trellis[i,:, t-1]) ).argmax(0)
                #print 'at time t=%d' %t
                #print 'state: ', state[i,t]
                #print 'action: ', action[i,t]
                #print 'trellis: ', trellis[i,:,t]
                #print 'backpt: ', backpt[i,t]

            # termination
            tokens = [trellis[i,:, -1].argmax()]
            for j in xrange(action.shape[1]-1, 0, -1):
                tokens.append(backpt[i, j])      
            
        backpt[:,0] =  self.N-1 # first state was missed
        #backpt = backpt -1   # represent state from -1 to N
        #tokens.append(self.N-1) # first state was missed                    
        return backpt    #np.array(tokens[::-1], 'int32' )
