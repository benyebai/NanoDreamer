# TinyDreamer

A small Dreamer-inspired learning project built on my [TensorGrad](https://github.com/benyebai/TensorGrad) and [MiniNN](https://github.com/benyebai/MiniNN) libraries.

- Project 5: a Pac-Man-style toy game, trajectories, returns, and actor-critic learning.
- Project 6: a visual encoder-decoder for MinAtar Space Invaders. MinAtar supplies 10×10 object-channel observations, not raw video pixels.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/python play.py                       # play the toy game
.venv/bin/python play.py --ghost               # play against a ghost
.venv/bin/python train_actor_critic.py         # train the toy-game actor-critic
.venv/bin/python play.py --ghost --actor       # watch a trained actor
.venv/bin/python train_visual.py               # train the Space Invaders encoder
.venv/bin/python -m unittest discover -s tests -v
```

Train before using `--actor` on a fresh clone. Actor checkpoints are saved in `checkpoints/` and ignored by Git. Arrow keys or WASD move; R restarts. Game sprites: [Genjoyers/Pac-Man](https://github.com/Genjoyers/Pac-Man), [Apache 2.0](assets/LICENSE).

## Layout

- `tinydreamer/`: game, actor, critic, trajectory, returns, evaluation, and visual encoder.
- `train.py`, `train_actor_critic.py`, `train_visual.py`: training entry points.
- `play.py`: interactive toy game and policy playback.
- `tests/`: game, learning, and encoder checks.

## Space Invaders encoder result

The simple experiment uses 300 training episodes (16,974 frames) and 20 separate test episodes (1,024 frames). The encoder compresses each 600-value observation to 64 values and the decoder reconstructs it. After 10,000 updates, the test foreground F1 was **0.993** using a fixed 0.5 cutoff.

See [`artifacts/visual/metrics.json`](artifacts/visual/metrics.json) and [`artifacts/visual/reconstructions.png`](artifacts/visual/reconstructions.png). The image shows original observations on top and reconstructions below. This tests reconstruction of the current observation; it does not predict the next game state.
