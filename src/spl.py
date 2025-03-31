# src/spl.py
import numpy as np

class SelfPacedLearning:
    def __init__(self, lmbda=0.5, alpha=0.1):
        """
        Self-Paced Learning initialization.
        :param lmbda: Initial lambda value controlling difficulty
        :param alpha: Increment for lambda per epoch
        """
        self.lmbda = lmbda
        self.alpha = alpha

    def compute_sample_weights(self, losses):
        """
        Compute sample weights based on the current lambda and loss values.
        :param losses: Array of loss values for each sample
        :return: Sample weights (1 if loss < lambda else 0)
        """
        weights = np.where(losses < self.lmbda, 1.0, 0.0)
        # Update lambda for the next epoch to gradually increase difficulty
        self.lmbda += self.alpha
        return weights
