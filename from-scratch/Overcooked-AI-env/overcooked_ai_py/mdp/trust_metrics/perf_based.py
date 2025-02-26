from collections import deque
from overcooked_ai_py.mdp.constants import *

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

    def to_dict(self, pos, orientation, held_obj):
        # update obs space 
        pass 