import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crazyflow_env import CrazyflowTrajectoryEnv

if __name__ == "__main__":
    print("Initializing Training Env...")
    
    train_env = DummyVecEnv([lambda: CrazyflowTrajectoryEnv(max_steps=500, target_speed=1.0, render_mode=None)])
    
    # Normalize observations and rewards
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        device="cpu" 
    )

    print("Starting Training...")
    # Train for 300k steps
    model.learn(total_timesteps=300_000) 
    
    # Save 
    print("Saving model and stats...")
    model.save("ppo_crazyflow_tracker")
    train_env.save("vec_normalize.pkl")
    
    print("Training finished.")
    train_env.close()