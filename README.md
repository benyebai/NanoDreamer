# TinyDreamer

Step 1:
A small Pac-Man-style game for learning actor-critic reinforcement learning.
The game uses Pygame, the actor and critic use my [TensorGrad](https://github.com/benyebai/TensorGrad) and [MiniNN](https://github.com/benyebai/MiniNN) libraries.

Next is the world model.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python play.py                  # play yourself
.venv/bin/python play.py --ghost          # play against a ghost
.venv/bin/python train_actor_critic.py    # train without a ghost
.venv/bin/python train_actor_critic.py --ghost  # train with a ghost
.venv/bin/python play.py --ghost --actor  # watch the trained actor
.venv/bin/python -m unittest discover -s tests -v
```

Train before using `--actor` on a fresh clone. Checkpoints are saved locally in
`checkpoints/` and ignored by Git. Arrow keys or WASD move; R restarts.

Sprites: [Genjoyers/Pac-Man](https://github.com/Genjoyers/Pac-Man), [Apache 2.0](assets/LICENSE).
