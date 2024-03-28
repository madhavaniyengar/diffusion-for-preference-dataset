from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
import sys
sys.path.append('/Users/sagarpatil/sagar/projects/diffusion-features/')
# from minigrid.manual_control import ManualControl
from environment.envs.manual_control import ManualControl
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
        self.agent_start_dir = np.random.randint(0, 4)
        self.size = size

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
    
    def reset(self, seed=None):
        super().reset()
        self.agent_start_pos = (np.random.randint(1, self.size-1), np.random.randint(1, self.size-1))

    def astar_path(self, start, goal):
        """A* algorithm to find the path between the start and goal positions."""
        # get the grid
        grid = self.grid.encode()
        # get the start and goal positions
        start = tuple(start)
        goal = tuple(goal)
        # get the width and height of the grid
        print(grid.shape)
        width, height, _ = grid.shape
        # get the directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # get the cost of moving in each direction
        costs = [1, 1, 1, 1]
        # get the number of directions
        num_directions = len(directions)
        # get the number of cells
        num_cells = width * height
        # get the start and goal indices
        start_index = start[0] * width + start[1]
        goal_index = goal[0] * width + goal[1]
        # get the heuristic
        def heuristic(index):
            x = index // width
            y = index % width
            return abs(x - goal[0]) + abs(y - goal[1])
        # get the g score
        g_score = np.inf * np.ones(num_cells)
        g_score[start_index] = 0
        # get the f score
        f_score = np.inf * np.ones(num_cells)
        f_score[start_index] = heuristic(start_index)
        # get the open set
        open_set = set([start_index])
        # get the closed set
        closed_set = set()
        # get the came from
        came_from = np.zeros(num_cells, dtype=int)
        # get the path
        path = []
        # get the current index
        current_index = start_index
        # get the current position
        current = start
        # get the current f score
        current_f_score = f_score[current_index]
        # get the current g score
        current_g_score = g_score[current_index]
        # get the current h score
        current_h_score = current_f_score - current_g_score
        # get the goal reached flag
        goal_reached = False
        # loop until the open set is empty
        while open_set:
            # get the current index
            current_index = min(open_set, key=lambda index: f_score[index])
            # get the current position
            current = (current_index // width, current_index % width)
            # check if the goal is reached
            if current == goal:
                goal_reached = True
                break
            # remove the current index from the open set
            open_set.remove(current_index)
            # add the current index to the closed set
            closed_set.add(current_index)
            # loop through the directions
            for i in range(num_directions):
                # get the next position
                next = (current[0] + directions[i][0], current[1] + directions[i][1])
                # check if the next position is valid
                if next[0] < 0 or next[0] >= height or next[1] < 0 or next[1] >= width:
                    continue
                # get the next index
                next_index = next[0] * width + next[1]
                # check if the next index is in the closed set
                if next_index in closed_set:
                    continue
                # get the next g score
                next_g_score = current_g_score + costs[i]
                # check if the next index is not in the open set
                if next_index not in open_set:
                    open_set.add(next_index)
                # check if the next g score is greater than the next index g score
                if next_g_score >= g_score[next_index]:
                    continue
                # update the g score
                g_score[next_index] = next_g_score
                # update the f score
                f_score[next_index] = g_score[next_index] + heuristic(next_index)
                # update the came from
                came_from[next_index] = current_index
        # check if the goal is reached
        if goal_reached:
            # get the current index
            current_index = goal_index
            # loop until the current index is the start index
            while current_index != start_index:
                # get the current position
                current = (current_index // width, current_index % width)
                # add the current position to the path
                path.append(current)
                # get the current index
                current_index = came_from[current_index]
            # add the start position to the path
            path.append(start)
            # reverse the path
            path = path[::-1]
        return path
        
    def distance(self, start, goal):
        """Calculate the distance between the start and goal positions."""
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def main():
    #env = SimpleEnv(render_mode="human")
    # env = gymnasium.make("MiniGrid-LavaGapS7-v0", render_mode="human")
    env = LavaEnv(render_mode="human")
    env.mission = "Walking away from lava towards the goal."
    # env_obs = FullyObsWrapper(env)
    # obs, _ = env_obs.reset()
    env.reset()
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    # print(obs['image'][1, 1, :])
    # print(obs['image'].shape)

    
if __name__ == "__main__":
    main()