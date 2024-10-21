from gymnasium.spaces import Box
from gymnasium import ObservationWrapper
import numpy as np
import torch

class AutoEncoderWrapper(ObservationWrapper):
    def __init__(self, env, scaler):
        super(AutoEncoderWrapper, self).__init__(env)
        self.scaler = scaler
        
    def observation(self, observation):
        # Convert observation to vector
        # Scale the observation
        obs_scaled = self.scaler.transform([observation])
        return obs_scaled
