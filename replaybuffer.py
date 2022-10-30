import numpy as np

class ReplayBuffer(): #
    def __init__(self, length = 10000):
    
        # Buffer Collection: (A, S, S',R , log_prob (if available),  D)
        # Done represents a mask of either 0 and 1
        self.length = length
        self.buffer = []

    def add(self, sample):
        if (len(self.buffer) > self.length):
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size):
        
        idx = np.random.permutation(len(self.buffer))[:batch_size]
        state_b = []
        action_b = []
        reward_b = []
        nextstate_b = []
        done_b = [] 
        log_prob = []
        for i in idx:
            if (len(self.buffer[0])==5):
                a, s, sp, r, d = self.buffer[i]
            else:
                a, s, sp, r, d, lp = self.buffer[i]
                log_prob.append(lp)
            state_b.append(s)
            action_b.append(a)
            reward_b.append(r)
            nextstate_b.append(sp)
            done_b.append(d)

        state_b = np.array(state_b)
        action_b = np.array(action_b)
        reward_b = np.array(reward_b)
        nextstate_b = np.array(nextstate_b)
        done_b = np.array(done_b)

        if len(self.buffer[0]) == 5:
            return (action_b, state_b, nextstate_b, reward_b, done_b)
        else:
            return (action_b, state_b, nextstate_b, reward_b, done_b, np.array(log_prob))