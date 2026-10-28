# NanoDreamer

The final agent from [my blog](https://www.benjamin-bai.com/projects/nanodreamer).
A CNN encodes each frame, a transformer predicts what happens next, and an actor
learns from five-step imagined rollouts. The critic uses real and imagined returns.

```text
nanodreamer/
  config.py     # training settings
  models.py     # encoder, decoder and world model
  agent.py      # actor, critic and learning updates
  replay.py     # episode storage and sampling
  losses.py     # distributional losses and lambda returns
scripts/
  train.py      # gameplay, training, evaluation and checkpoints
tests/
  test_agent.py
  test_training.py
```

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m scripts.train space_invaders artifacts/space_invaders --seed 0
```

Defaults: 4k random prefill actions, 500k policy actions, CUDA or CPU. Final
evaluation uses 100 games, seed 950000 and a 1,000-action cap; intermediate
checks use a 200-action cap. A time limit can stop training early; `result.json`
records the actual steps. Use `--resume` to continue a run. The blog used seed 1
for the other four games.

Earlier versions and experiments are preserved in ignored `archive/`.
