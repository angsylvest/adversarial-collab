from collections import deque
from overcooked_ai_py.mdp.constants import *
import numpy as np 
import math 

class ShannonEntropy: 
    # update to include params relevant to keep track of 

    def __init__(self, sliding_window=False, num_collected=None, window = None, num_unexpected=None):
        self.sliding_window = sliding_window 
        
        if self.num_collected:
            self.num_collected = num_collected 
        else: 
            self.num_collected = 0 
        
        if num_collected: 
            self.num_unexpected = num_unexpected
        else: 
            self.num_unexpected = 0 

        if self.sliding_window: 
            if not window: 
                self.window = deque(maxlen=100)
            else: 
                self.window = window


        self.entropy = self.adaptively_discretize_trust(self.calc_entropy())
        
        # if not using sliding window, will just average everything 
        # if using sliding window, will only average most recent 100

    def update(self, interact_info):
        lazy = interact_info["lazy"]
        adversarial = interact_info["adversarial"]

        if lazy or adversarial: 
            self.window.append("unexpected")
            self.num_unexpected
        else: 
            self.window.append("expected")

        self.num_collected += 1
        
        self.entropy = self.adaptively_discretize_trust(self.calc_entropy())
        
        # might be able to split into entropy of each as well .. 

    def one_hot(self, index, num_bins):
        """
        Converts an index value to a one-hot encoded vector.
        """
        one_hot = np.zeros(num_bins)
        one_hot[index] = 1
        return one_hot
    

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
        
    def calc_entropy(self): 
        if not self.window:
            return 0  # Avoid log(0) issues when the window is empty
        
        prob_stable = self.window.count("expected") / len(self.window)
        prob_unstable = 1 - prob_stable  # Probability of acting unexpectedly

        # Avoid log(0) issues by only computing entropy for nonzero probabilities
        entropy = 0
        if prob_stable > 0:
            entropy -= prob_stable * math.log2(prob_stable)
        if prob_unstable > 0:
            entropy -= prob_unstable * math.log2(prob_unstable)

        return entropy
    
    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 