import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maximizes the Q-value 
        # at the current observation as the output
        Qs = self.critic.qa_values(obs=obs)
        action = np.argmax(Qs, axis=0)

        print("Check ArgMaxPolicy shape!")

        return action.squeeze()
