from collections import deque

import numpy as np

from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line, draw_points


def main():
    fps = 30
    rgbas = np.random.default_rng(0).uniform(0, 1, (1, 4))
    rgbas[..., 3] = 1.0

    sim = Sim(control="state")
    sim.reset()
    # Replace the position of the first drone in the first world. JAX arrays are immutable, which is
    # why we cannot change the sim.data object in-place. Instead, we need to create a new sim.data
    # object with the desired changes by calling sim.data.replace(). The same logic applies to the
    # sim.data.states object contained within sim.data, and the sim.data.states.pos array. For more
    # information on changing JAX arrays, see:
    # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
    # Define three waypoints (x, y, z)
    waypoints = np.array(
        [
            np.array([1.0, 1.0, 0.2]),
            np.array([0.0, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.8]),
            np.array([1.0, 1.0, 0.2]),
        ]
    )

    # set initial position to first waypoint
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[0, 0].set(waypoints[0]))
    )

    control = np.zeros((sim.n_worlds, sim.n_drones, 13))

    # duration per waypoint
    steps_per_waypoint = 5 * sim.control_freq

    pos = deque()

    for wp in waypoints:
        control[..., :3] = wp
        for i in range(steps_per_waypoint):
            sim.state_control(control)
            sim.step(sim.freq // sim.control_freq)
            # sim.render()

            if i % 20 == 0:
                pos.append(sim.data.states.pos[0, :])
            if ((i * fps) % sim.control_freq) < fps:
                lines = np.array(pos)
                draw_line(sim, lines[:, 0, :], rgbas[0, :], start_size=0.3, end_size=3.0)
                draw_points(sim, points=waypoints)
                sim.render()

    sim.close()


if __name__ == "__main__":
    main()
