import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import crazyflow as cf
from collections import deque
from gymnasium import spaces
from crazyflow.sim.visualize import draw_line, draw_points

class CrazyflowTrajectoryEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_steps=1000, target_speed=1.0, render_mode=None):
        super(CrazyflowTrajectoryEnv, self).__init__()

        self.max_steps = max_steps
        self.target_speed = target_speed
        self.render_mode = render_mode
        self.current_step = 0

        # Using thrust control (raw motors).
        self.sim = cf.Sim(control="thrust")
        self.rng = jax.random.PRNGKey(0)

        self.steps_per_env_step = self.sim.freq // self.sim.control_freq
        self.dt = 1.0 / self.sim.control_freq

        # Visualization setup
        self.path_history = deque(maxlen=500)
        self.path_color = np.array([1.0, 0.0, 0.0, 1.0])  # Red path

        # PPO outputs [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Obs: [Quat(4), AngVel(3), LinVel(3), RelPos(3), RelVel(3)] = 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.trajectory_points = self._generate_circle_trajectory()

    def _generate_circle_trajectory(self):
        t = np.linspace(0, self.max_steps * self.dt, self.max_steps)
        radius = 1.0
        omega = self.target_speed / radius

        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = np.ones_like(t) * 1.0

        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = np.zeros_like(t)

        pos = np.stack([x, y, z], axis=1)
        vel = np.stack([vx, vy, vz], axis=1)
        return pos, vel

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.path_history.clear()

        self.sim.reset()

        # Start exactly at the first target point to help exploration
        start_pos = self.trajectory_points[0][0]

        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=self.sim.data.states.pos.at[0, 0].set(start_pos),
                vel=self.sim.data.states.vel.at[0, 0].set(np.zeros(3)),
                ang_vel=self.sim.data.states.ang_vel.at[0, 0].set(np.zeros(3)),
                # Reset orientation to upright (id. quat.)
                quat=self.sim.data.states.quat.at[0, 0].set(np.array([0.0, 0.0, 0.0, 1.0])),
            )
        )

        return self._get_obs(), {}

    def step(self, action):
        # [-1, 1] -> [0.0, 0.16]
        # Hover is ~0.075 => allow full range now for agility.
        thrust_cmd = (np.array(action) + 1.0) * 0.08
        thrust_cmd = np.clip(thrust_cmd, 0.0, 0.16)

        control_input = np.zeros((self.sim.n_worlds, self.sim.n_drones, 4))
        control_input[0, 0] = thrust_cmd

        # thrust_control works. The API should not change
        self.sim.thrust_control(control_input)
        # if hasattr(self.sim, "thrust_control"):
        #     self.sim.thrust_control(control_input)
        # else:
        #     self.sim.data = self.sim.data.replace(inputs=jnp.array(control_input))

        self.sim.step(self.steps_per_env_step)
        self.current_step += 1

        if self.render_mode == "human":
            current_pos = np.array(self.sim.data.states.pos[0, 0])
            self.path_history.append(current_pos)

        traj_idx = min(self.current_step, len(self.trajectory_points[0]) - 1)
        target_pos = self.trajectory_points[0][traj_idx]
        target_vel = self.trajectory_points[1][traj_idx]

        curr_pos = np.array(self.sim.data.states.pos[0, 0])
        curr_vel = np.array(self.sim.data.states.vel[0, 0])

        dist = np.linalg.norm(curr_pos - target_pos)
        vel_dist = np.linalg.norm(curr_vel - target_vel)

        # reward func.
        r_survival = 0.1
        r_pos = np.exp(-2.0 * dist**2)
        r_vel = 0.1 * np.exp(-0.5 * vel_dist**2)
        r_stable = -0.05 * np.linalg.norm(self.sim.data.states.ang_vel[0, 0]) ** 2

        reward = r_survival + r_pos + r_vel + r_stable

        terminated = False
        truncated = False

        # Crash conditions
        if curr_pos[2] < 0.05 or np.linalg.norm(curr_pos) > 5.0:
            terminated = True
            reward = -10.0

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        traj_idx = min(self.current_step, len(self.trajectory_points[0]) - 1)
        target_pos = self.trajectory_points[0][traj_idx]
        target_vel = self.trajectory_points[1][traj_idx]

        pos = np.array(self.sim.data.states.pos[0, 0])
        vel = np.array(self.sim.data.states.vel[0, 0])
        quat = np.array(self.sim.data.states.quat[0, 0])
        ang_vel = np.array(self.sim.data.states.ang_vel[0, 0])

        rel_pos = target_pos - pos
        rel_vel = target_vel - vel

        obs = np.concatenate([quat, ang_vel, vel, rel_pos, rel_vel])
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode != "human":
            return

        if len(self.path_history) > 1:
            points = np.array(self.path_history)
            draw_line(self.sim, points, self.path_color, start_size=0.01, end_size=0.05)

        traj_idx = min(self.current_step, len(self.trajectory_points[0]) - 1)
        target = self.trajectory_points[0][traj_idx]
        draw_points(self.sim, points=target[None, :])

        self.sim.render()

    def close(self):
        if hasattr(self, "sim"):
            self.sim.close()
        super().close()