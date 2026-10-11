# The rules for our small Pac-Man-style game.

import numpy as np

LEVEL = [
    "#######",
    "#P...##",
    "#.#...#",
    "#...#.#",
    "##....#",
    "#.....#",
    "#######",
]

# Actions: 0=up, 1=right, 2=down, 3=left.
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GHOST_START = (5, 5)


class PelletMaze:
    def __init__(self, max_steps=80, with_ghost=False, full_view=True):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.max_steps = max_steps
        self.with_ghost = with_ghost
        self.full_view = full_view
        self.action_count = 4
        if with_ghost and full_view:
            self.observation_shape = (len(LEVEL), len(LEVEL[0]), 4)
        else:
            self.observation_shape = (3, 3, 4 if with_ghost else 3)

        self.floor = set()
        self.start = None
        self.starting_pellets = set()
        for row in range(len(LEVEL)):
            for col in range(len(LEVEL[row])):
                tile = LEVEL[row][col]
                if tile != "#":
                    self.floor.add((row, col))
                if tile == ".":
                    self.starting_pellets.add((row, col))
                if tile == "P":
                    self.start = (row, col)

        self.started = False

    def reset(self):
        self.player = self.start
        self.ghost = GHOST_START if self.with_ghost else None
        self.pellets = self.starting_pellets.copy()
        self.steps = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.won = False
        self.caught = False
        self.started = True
        return self.observe(), self.info()

    def observe(self):
        if not self.started:
            raise RuntimeError("Call reset() first")

        # Channels: wall, pellet, player, and ghost.
        view = np.zeros(self.observation_shape, dtype=np.float32)
        if self.with_ghost and self.full_view:
            # Full-view ghost mode shows the entire 7 x 7 maze.
            for row in range(len(LEVEL)):
                for col in range(len(LEVEL[row])):
                    place = (row, col)
                    if place not in self.floor:
                        view[row, col, 0] = 1
                    elif place in self.pellets:
                        view[row, col, 1] = 1
            view[self.player[0], self.player[1], 2] = 1
            view[self.ghost[0], self.ghost[1], 3] = 1
        else:
            # The original game and the older ghost actor use a 3 x 3 view.
            for view_row in range(3):
                for view_col in range(3):
                    row = self.player[0] + view_row - 1
                    col = self.player[1] + view_col - 1
                    place = (row, col)
                    if place not in self.floor:
                        view[view_row, view_col, 0] = 1
                    elif place in self.pellets:
                        view[view_row, view_col, 1] = 1
                    if self.with_ghost and place == self.ghost:
                        view[view_row, view_col, 3] = 1
            view[1, 1, 2] = 1
        return view

    def step(self, action):
        if not self.started:
            raise RuntimeError("Call reset() first")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is over. Call reset()")
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be 0, 1, 2, or 3")

        row_change, col_change = MOVES[action]
        new_place = (self.player[0] + row_change, self.player[1] + col_change)
        hit_wall = new_place not in self.floor
        ate_pellet = False
        reward = -0.01

        if hit_wall:
            reward -= 0.05
        else:
            self.player = new_place

        self.steps += 1
        if self.with_ghost and self.player == self.ghost:
            self.caught = True
        else:
            if self.player in self.pellets:
                self.pellets.remove(self.player)
                ate_pellet = True
                reward += 1.0

            if len(self.pellets) == 0:
                self.won = True
                reward += 2.0
            elif self.with_ghost and self.steps % 3 == 0:
                self.move_ghost()
                if self.player == self.ghost:
                    self.caught = True

        if self.caught:
            reward -= 5.0
        self.terminated = self.won or self.caught
        if self.steps >= self.max_steps and not self.terminated:
            self.truncated = True

        self.total_reward += reward
        details = self.info()
        details["blocked"] = hit_wall
        details["collected"] = ate_pellet
        return self.observe(), reward, self.terminated, self.truncated, details

    def move_ghost(self):
        """Every third turn, move one square closer to Pac-Man."""
        choices = []
        for row_change, col_change in MOVES:
            place = (self.ghost[0] + row_change, self.ghost[1] + col_change)
            if place in self.floor:
                distance = abs(place[0] - self.player[0]) + abs(
                    place[1] - self.player[1]
                )
                choices.append((distance, place))
        if choices:
            self.ghost = min(choices)[1]

    def info(self):
        return {
            "pellets_remaining": len(self.pellets),
            "steps": self.steps,
            "total_reward": self.total_reward,
            "won": self.won,
            "caught": self.caught,
        }

    def snapshot(self):
        """Save the game so we can replay the same situation later."""
        if not self.started:
            raise RuntimeError("Call reset() first")
        return {
            "player": self.player,
            "ghost": self.ghost,
            "pellets": self.pellets.copy(),
            "steps": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "won": self.won,
            "caught": self.caught,
        }

    def restore(self, saved):
        """Put the game back at a saved situation."""
        if saved["player"] not in self.floor:
            raise ValueError("Player is outside the maze")
        if not saved["pellets"].issubset(self.starting_pellets):
            raise ValueError("A pellet is outside the maze")
        if self.with_ghost and saved["ghost"] not in self.floor:
            raise ValueError("Ghost is outside the maze")
        self.player = saved["player"]
        self.ghost = saved["ghost"]
        self.pellets = saved["pellets"].copy()
        self.steps = saved["steps"]
        self.total_reward = saved["total_reward"]
        self.terminated = saved["terminated"]
        self.truncated = saved["truncated"]
        self.won = saved["won"]
        self.caught = saved["caught"]
        self.started = True

    def render(self):
        """A text version of the maze, useful when debugging."""
        if not self.started:
            raise RuntimeError("Call reset() first")
        rows = []
        for row in range(len(LEVEL)):
            line = ""
            for col in range(len(LEVEL[row])):
                place = (row, col)
                if place == self.player and place == self.ghost:
                    line += "X"
                elif place == self.player:
                    line += "P"
                elif place == self.ghost:
                    line += "G"
                elif LEVEL[row][col] == "#":
                    line += "#"
                elif place in self.pellets:
                    line += "."
                else:
                    line += " "
            rows.append(line)
        return "\n".join(rows)
