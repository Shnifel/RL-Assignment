from A2C_env import Gym2OpEnv
import numpy as np
from sklearn.preprocessing import StandardScaler
from ae import AutoEncoder
import joblib
from tqdm import tqdm

def collect_observations(n_obs = 5e5):
    env = Gym2OpEnv()
    obs = env.reset()

    observations = []
    num_steps = int(n_obs)

    for _ in tqdm(range(num_steps)):
        action = env.action_space.sample()  
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs) 
        if terminated:
            obs = env.reset()
    
    return observations

def train_ae(observations):
    observations_array = np.array(observations)
    scaler = StandardScaler()
    observations = scaler.fit_transform(observations_array)
    joblib.dump(scaler, 'scaler.save')
    # auto_encoder = AutoEncoder(observations_array.shape[1], 32)
    # auto_encoder.train_ae(observations, n_epochs=15, lr=1e-4)

if __name__ == "__main__":
    observations = collect_observations()
    train_ae(observations)


