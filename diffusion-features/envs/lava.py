from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
# from minigrid.manual_control import ManualControl
from manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium
from minigrid.wrappers import FullyObsWrapper



class LavaEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        if agent_start_pos is not None:
            self.agent_start_pos = agent_start_pos
        else:
            # randomly start the agent
            self.agent_start_pos = (np.random.randint(1, size-1), np.random.randint(1, size-1))
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())
        
        # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place lava randomly in the environment
        random_coords = np.random.randint(1, width-1, size=2)
        self.put_obj(Lava(), random_coords[0], random_coords[1])
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # self.mission = "grand mission"


def main():
    #env = SimpleEnv(render_mode="human")
    # env = gymnasium.make("MiniGrid-LavaGapS7-v0", render_mode="human")
    env = LavaEnv(render_mode="human")
    env.mission = "Walking away from lava towards the goal."
    env_obs = FullyObsWrapper(env)
    obs, _ = env_obs.reset()
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42, save_path="../data/lavaenv/manual_control1.txt")
    manual_control.start()
    # print(obs['image'][1, 1, :])
    # print(obs['image'].shape)

    
if __name__ == "__main__":
    main()