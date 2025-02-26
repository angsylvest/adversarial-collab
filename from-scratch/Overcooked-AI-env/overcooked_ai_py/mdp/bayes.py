import numpy as np 
from overcooked_ai_py.mdp.constants import *

class MultiTravosBayes: 

    def __init__(self, num_interactive):
        self.num_interactive = num_interactive

        self.alpha_lazy = np.ones(self.num_interactive)
        self.beta_lazy = np.ones(self.num_interactive)
        self.trust_lazy = np.full(self.num_interactive, 0.5)
        self.uncertainty_lazy = self.calc_uncertainty(self.alpha_lazy, self.beta_lazy)

        self.alpha_adv = np.ones(self.num_interactive)
        self.beta_adv = np.ones(self.num_interactive)
        self.trust_adv = np.full(self.num_interactive, 0.5)
        self.uncertainty_adv = self.calc_uncertainty(self.alpha_adv, self.beta_adv)

    def update_bayes(self, type, success=False, indx=0):
        if type ==  "adversarial": 
            self.update_adv_trust(success, indx) 
        else: 
            self.update_lazy_trust(success, indx)

    def update_lazy_trust(self, success=False, encounter_indx=0):
        if success:
            self.beta_lazy[encounter_indx] += 1  # Increment beta on success
        else:
            self.alpha_lazy[encounter_indx] += 1  # Increment alpha on failure
        
        # Update trust for this index
        self.trust_lazy[encounter_indx] = self.beta_lazy[encounter_indx] / (
            self.alpha_lazy[encounter_indx] + self.beta_lazy[encounter_indx]
        )

        self.uncertainty_lazy = self.calc_uncertainty(self.alpha_lazy, self.beta_lazy)

        if discretize_trust:
            self.trust_lazy = self.adaptively_discretize_trust(self.trust_lazy)
            self.uncertainty_lazy = self.adaptively_discretize_trust(self.uncertainty_lazy)
        
    def update_adv_trust(self, success=False, encounter_indx=0):
        if success:
            self.beta_adv[encounter_indx] += 1  # Increment beta on success
        else:
            self.alpha_adv[encounter_indx] += 1  # Increment alpha on failure
        
        # Update trust for this index
        self.trust_adv[encounter_indx] = self.beta_adv[encounter_indx] / (
            self.alpha_adv[encounter_indx] + self.beta_adv[encounter_indx]
        )

        self.uncertainty_adv = self.calc_uncertainty(self.self.alpha_adv, self.beta_adv)

        if discretize_trust:
            self.trust_adv = self.adaptively_discretize_trust(self.trust_adv)
            self.uncertainty_adv = self.adaptively_discretize_trust(self.uncertainty_adv)
    
    def calc_uncertainty(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))


    def adaptively_discretize_trust(self, trust_score, adapt=False):

        if trust_score > 0.98 and adapt:
            # Focus on fine-grained changes for high trust
            bins = [0.99, 0.995, 0.998, 1.0]
        else:
            # Broader bins for lower trust
            # bins = [0, 0.5, 0.8, 0.95, 0.99]
            bins = [0, 0.25, 0.5, 0.75, 0.99]
        return np.digitize([trust_score], bins)