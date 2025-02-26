from collections import deque
from overcooked_ai_py.mdp.constants import *

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
        
        # might be able to split into entropy of each as well .. 

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 