from collections import deque
from overcooked_ai_py.mdp.constants import *
import numpy as np 

class PerfBased: 
    # update to include params relevant to keep track of 

    def __init__(self, multi=False, sliding_window=False):
        self.sliding_window = sliding_window 
        self.num_collected = 0 
        # if not using sliding window, will just average everything 
        # if using sliding window, will only average most recent 100 

        if self.sliding_window: 
            self.window = deque(maxlen=100)

    def update(self, interact_info):
        lazy = interact_info["lazy"]
        adversarial = interact_info["adversarial"]

        # use to calc trust + uncertainity

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

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 