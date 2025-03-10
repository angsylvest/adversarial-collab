from collections import deque
from overcooked_ai_py.mdp.constants import *
import numpy as np 

class PerfBased: 
    # update to include params relevant to keep track of 

    def __init__(self, multi=False, sliding_window=False):
        self.sliding_window = sliding_window 
        self.num_collected = 0 
        self.multi = multi

        # if self.sliding_window: 
        if not multi: 
            self.trust_lazy = deque(maxlen=100)

        else: 
            self.trust_lazy = deque(maxlen=100)
            self.trust_adversarial = deque(maxlen=100)

        self.lazy_trust, self.lazy_uncert = self.calc_trust(self.trust_lazy)
        self.adversarial_trust, self.adversarial_uncert = self.calc_trust(self.trust_adversarial) 

    def update(self, interact_info):
        lazy = interact_info["lazy"]
        adversarial = interact_info["adversarial"]

        # use to calc trust + uncertainity
        if self.multi: 
            if lazy: 
                self.trust_lazy.append(1) 
            if not lazy: 
                self.trust_lazy.append(0) 
            if adversarial: 
                self.trust_adversarial.append(1)  
            if not adversarial: 
                self.trust_adversarial.append(0)

            # update corresponding trust for each 
            self.lazy_trust, self.lazy_uncert = self.calc_trust(self.trust_lazy)
            self.adversarial_trust, self.adversarial_uncert = self.calc_trust(self.trust_adversarial) 

        else: 
            if lazy or adversarial:
                self.trust_lazy.append(1) 
            else: 
                self.trust_lazy.append(0) 

            # update corresponding trust for each 
            self.lazy_trust, self.lazy_uncert = self.calc_trust(self.trust_lazy)


    def one_hot(self, index, num_bins):
        """
        Converts an index value to a one-hot encoded vector.
        """
        one_hot = np.zeros(num_bins)
        one_hot[index] = 1
        return one_hot
    
    
    def calc_trust(self, trust_list): 
        beta = trust_list.count(1)
        alpha = trust_list.count(0)

        trust_score = alpha / (alpha + beta)
        uncert = self.calc_uncertainty(alpha, beta)

        return trust_score, uncert
    

    def calc_uncertainty(self, alpha, beta): 
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    def adaptively_discretize_trust(self, trust_score, adapt=False):

        if trust_score > 0.98 and adapt:
            # Focus on fine-grained changes for high trust
            bins = [0.99, 0.995, 0.998, 1.0]
        else:
            # Broader bins for lower trust
            # bins = [0, 0.5, 0.8, 0.95, 0.99]
            bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        index = np.digitize([trust_score], bins, right=True)[0] 
        if one_hot_encode:
            return self.one_hot(index-1, len(bins))
            
        else: 
            return index
