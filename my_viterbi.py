
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
        backpt = np.ones(trellis.shape, 'int32') * -1
        tokens = np.ones(action.shape, 'int32') * -1

        # initialization at t=0 we are at the highest state        
        trellis[:,self.N-1, 0] = 1
        
        
        for i in xrange(0, action.shape[0]):
            for t in xrange(1, action.shape[1]):
                
                if (reward[i,t]<action[i,t]):
                    # Failure state
                    # the current and previous states are well known in this case
                    
                    # update current state to be fail
                    trellis[i,:,t] = np.zeros(self.N)
                    trellis[i,0,t] = 1.0
                    # update previous state according to the reward
                    # if reward=0 take the most probable between 0 and F states
                    if (reward[i,t] == 0):
                        backpt[i,:, t] = trellis[i,0:2,t-1].argmax()
                    else:
                        backpt[i,0, t] = reward[i,t]+1
                else:
                    # old
                    #trellis[i,:, t] = transProb[action[i,t]].dot(trellis[i,:, t-1])     # max 0 is the biggest from each col.
                    #backpt[i, t] = ( transProb[action[i,t]].dot(trellis[i,:, t-1]) ).argmax(0)                    
                    
                    # NEW
                    #bb = np.expand_dims(trellis[i,:, t-1], axis=0)
                    #transProb[action[i,t]].dot(bb.T)
                    
                    trellis[i,:, t] = (trellis[i,:, t-1]*transProb[action[i,t]]).max(1)
                    #backpt[i,:,t] = (np.tile(trellis[i,:, t-1], [self.N,1]) * transProb[action[i,t]]).argmax(1)
                    # use random choice in order to avoid that first element is always selected
                    backpt[i,:,t] = (np.tile(trellis[i,:, t-1], [self.N,1]) * transProb[action[i,t]]).argmax(1)
                #print 'at time t=%d' %t
                #print 'state: ', state[i,t]
                #print 'action: ', action[i,t]
                #print 'trellis: ', trellis[i,:,t]
                #print 'backpt: ', backpt[i,t]           
            
            # termination
            tokens[i,0] = trellis[i,:, -1].argmax()
            
            for j in xrange(1, backpt.shape[2]):
                tokens[i,j] = backpt[i,tokens[i,j-1],-(j+1)]
        
        
        # reverse to get the state in the forward direction
        tokens = np.flip(tokens, 1)        
        # shift by 1
        tokens = np.roll(tokens,-1)
        return (tokens, trellis, backpt)
                
            