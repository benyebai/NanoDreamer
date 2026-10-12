import json
from pathlib import Path

import numpy as np
import pygame
from minatar import Environment
from mininn.checkpoint import save_checkpoint
from mininn.optimizers import Adam, clip_grad_norm
from tensorgrad import Tensor

from tinydreamer.visual import (
    VisualAutoencoder,
    reconstruction_metrics,
    weighted_binary_cross_entropy,
)

TRAIN_EPISODES = 300
TEST_EPISODES = 20
STEPS = 10_000
BATCH_SIZE = 32
OUTPUT = Path("artifacts/visual")


def collect_frames(episodes, seed):
    game = Environment("space_invaders")
    game.seed(seed)
    random = np.random.default_rng(seed)
    frames = []

    for _ in range(episodes):
        game.reset()
        for _ in range(150):
            frames.append(game.state().astype(float))
            action = int(random.choice(game.minimal_action_set()))
            _, done = game.act(action)
            if done:
                break

    return np.array(frames)


def color_frame(frame):
    # give each object type a display color
    colors = [
        (65, 190, 255),  # cannon
        (235, 190, 65),  # alien
        (255, 145, 70),  # alien moving left
        (255, 210, 90),  # alien moving right
        (65, 255, 180),  # friendly bullet
        (255, 75, 160),  # enemy bullet
    ]
    image = np.full((10, 10, 3), (16, 20, 28), dtype=np.uint8)
    for channel in (1, 2, 3, 4, 5, 0):
        image[frame[:, :, channel].astype(bool)] = colors[channel]
    return image


def save_examples(frames, probabilities, path):
    # here we get the pictures for the four original/reconstruction pairs from the test episodes
    image = np.full((21, 43, 3), 35, dtype=np.uint8)
    for column, index in enumerate(np.linspace(0, len(frames) - 1, 4, dtype=int)):
        left = column * 11
        image[:10, left : left + 10] = color_frame(frames[index])
        image[11:21, left : left + 10] = color_frame(probabilities[index] >= 0.5)

    image = np.repeat(np.repeat(image, 18, axis=0), 18, axis=1)
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    pygame.image.save(surface, str(path))


def main():
    np.random.seed(0)
    train_frames = collect_frames(TRAIN_EPISODES, seed=0)
    test_frames = collect_frames(TEST_EPISODES, seed=10_000)

    model = VisualAutoencoder(hidden_size=128, latent_size=64)
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    random = np.random.default_rng(1)

    for step in range(STEPS):
        batch = train_frames[random.integers(len(train_frames), size=BATCH_SIZE)]
        targets = Tensor(batch)
        loss = weighted_binary_cross_entropy(model(targets), targets)

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}: loss {float(loss.data):.4f}")

    logits = model(Tensor(test_frames)).data
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    scores = reconstruction_metrics(logits, test_frames)
    scores["train_frames"] = len(train_frames)
    scores["test_frames"] = len(test_frames)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    save_examples(test_frames, probabilities, OUTPUT / "reconstructions.png")
    save_checkpoint(OUTPUT / "autoencoder.npz", model, optimizer)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
