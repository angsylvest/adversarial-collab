import numpy as np

class TRAVOS:
    def __init__(self):
        # Stores recommender history for reliability assessments
        self.recommender_history = {}
        # Stores direct interaction history with target agents
        self.direct_interactions = {}

    def update_recommender_history(self, recommender, is_correct):
        """Update history of a recommender based on recommendation correctness."""
        if recommender not in self.recommender_history:
            self.recommender_history[recommender] = {'correct': 0, 'total': 0}
        
        self.recommender_history[recommender]['total'] += 1
        if is_correct:
            self.recommender_history[recommender]['correct'] += 1

    def get_reliability(self, recommender):
        """Calculate reliability of a recommender as a probability."""
        if recommender in self.recommender_history:
            correct = self.recommender_history[recommender]['correct']
            total = self.recommender_history[recommender]['total']
            return correct / total if total > 0 else 0.5
        else:
            return 0.5  # Neutral reliability if no history available

    def update_direct_interaction(self, target_agent, outcome):
        """
        Update the direct interaction history for the target agent.
        Outcome should be 1 for positive, 0 for negative interactions.
        """
        if target_agent not in self.direct_interactions:
            self.direct_interactions[target_agent] = {'positive': 0, 'total': 0}
        
        self.direct_interactions[target_agent]['total'] += 1
        if outcome == 1:
            self.direct_interactions[target_agent]['positive'] += 1

    def get_direct_trust(self, target_agent):
        """Calculate direct trust based on past interactions with the target agent."""
        if target_agent in self.direct_interactions:
            positive = self.direct_interactions[target_agent]['positive']
            total = self.direct_interactions[target_agent]['total']
            return positive / total if total > 0 else 0.5
        else:
            return 0.5  # Neutral trust if no direct interaction history

    def trust_update(self, target_agent, recommendations):
        """
        Update the trust in target_agent by combining direct trust and third-party recommendations.
        
        Parameters:
        - target_agent: The agent being evaluated for trust.
        - recommendations: List of tuples (recommender, opinion), where:
            - recommender is the agent giving the opinion
            - opinion is 1 if they trust the target_agent, 0 otherwise.
        
        Returns:
        - Overall trust score for the target_agent.
        """
        # Prior trust level based on direct interactions
        direct_trust = self.get_direct_trust(target_agent)
        trust_numerator = direct_trust
        trust_denominator = 1

        for recommender, opinion in recommendations:
            reliability = self.get_reliability(recommender)
            weight = reliability / (1 - reliability + 0.001)  # Transform reliability into weight
            
            # Bayesian update using weighted recommendations
            trust_numerator += weight * opinion
            trust_denominator += weight

        # Final trust score for the target_agent
        overall_trust = trust_numerator / trust_denominator
        return overall_trust

# Example Usage
travos_model = TRAVOS()

# Updating recommender history (simulate past recommendation outcomes)
travos_model.update_recommender_history('agent1', True)
travos_model.update_recommender_history('agent1', False)
travos_model.update_recommender_history('agent2', True)
travos_model.update_recommender_history('agent2', True)

# Direct interactions with target_agent
travos_model.update_direct_interaction('target_agent', 1)  # Positive outcome
travos_model.update_direct_interaction('target_agent', 0)  # Negative outcome

# Recommendations from third-party agents about target_agent
recommendations = [
    ('agent1', 1),  # agent1 trusts target_agent
    ('agent2', 0)   # agent2 distrusts target_agent
]

# Calculate overall trust score for target_agent
overall_trust = travos_model.trust_update('target_agent', recommendations)
print(f"Overall Trust in target_agent: {overall_trust:.2f}")
