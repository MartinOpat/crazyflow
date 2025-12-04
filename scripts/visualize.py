import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from training_env import CrazyflowTrajectoryEnv

if __name__ == "__main__":
    model_path = "ppo_crazyflow_tracker"
    stats_path = "vec_normalize.pkl"

    if not os.path.exists(model_path + ".zip") or not os.path.exists(stats_path):
        print(f"Error: Could not find model ({model_path}.zip) or stats ({stats_path}).")
        print("Please run train_ppo.py first.")
        exit(1)

    print("Loading Normalization Stats...")
    # Load stats
    temp_env = DummyVecEnv([lambda: CrazyflowTrajectoryEnv(max_steps=10)])
    stats_env = VecNormalize.load(stats_path, temp_env)
    
    obs_mean = stats_env.obs_rms.mean
    obs_var = stats_env.obs_rms.var
    epsilon = 1e-8
    
    stats_env.close()
    
    print("Initializing Visualization Env...")
    # Create rendering
    viz_env = CrazyflowTrajectoryEnv(max_steps=1000, target_speed=1.0, render_mode="human")
    
    print("Loading Model...")
    model = PPO.load(model_path, device="cpu")
    
    print("Starting Loop...")
    obs, _ = viz_env.reset()
    try:
        while True:
            norm_obs = (obs - obs_mean) / np.sqrt(obs_var + epsilon)
            norm_obs = np.clip(norm_obs, -10.0, 10.0)
            
            # Predict
            action, _ = model.predict(norm_obs, deterministic=True)
            
            # Step
            obs, reward, done, truncated, info = viz_env.step(action)
            
            # Render
            viz_env.render()
            
            if done or truncated:
                print("Episode finished. Resetting...")
                obs, _ = viz_env.reset()
                
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        viz_env.close()