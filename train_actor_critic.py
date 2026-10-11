"""Train TinyDreamer's actor and critic on real game episodes."""

import argparse
from pathlib import Path

import numpy as np
from mininn.checkpoint import save_checkpoint
from mininn.optimizers import Adam, clip_grad_norm
from tensorgrad import Tensor

from actor import Actor
from critic import Critic
from evaluate import evaluate_policy
from game import PelletMaze
from returns import calculate_lambda_returns
from trajectory import collect_episode


def train_one_episode(
    actor, critic, game, actor_optimizer, critic_optimizer, discount, lam=1.0
):
    episode = collect_episode(game, actor.choose_action)

    next_values = []
    for step in episode:
        next_values.append(critic.predict(step["next_observation"]))

    bootstrap = 0.0
    if episode[-1]["truncated"]:
        bootstrap = next_values[-1]
    returns = calculate_lambda_returns(episode, next_values, discount, lam, bootstrap)

    actor_loss = Tensor(0.0)
    critic_loss = Tensor(0.0)
    for step, future_reward in zip(episode, returns):
        value = critic.forward(step["observation"])
        advantage = future_reward - float(value.data[0, 0])

        scores = actor.forward(step["observation"])
        log_probs = scores - scores.logsumexp(axis=1, keepdims=True)
        chosen_log_prob = log_probs[0, step["action"]]
        actor_loss = actor_loss - chosen_log_prob * advantage

        error = value[0, 0] - future_reward
        critic_loss = critic_loss + error * error

    actor_loss = actor_loss / len(episode)
    critic_loss = critic_loss / len(episode)

    actor.network.zero_grad()
    actor_loss.backward()
    clip_grad_norm(actor.parameters(), 5.0)
    actor_optimizer.step()

    critic.network.zero_grad()
    critic_loss.backward()
    clip_grad_norm(critic.parameters(), 5.0)
    critic_optimizer.step()

    total_reward = sum(step["reward"] for step in episode)
    return total_reward, float(actor_loss.data), float(critic_loss.data)


def train(episodes=500, seed=0, discount=0.99, lam=1.0, with_ghost=False, full_view=True):
    game = PelletMaze(with_ghost=with_ghost, full_view=full_view)
    input_size = int(np.prod(game.observation_shape))
    actor = Actor(seed, input_size=input_size)
    critic = Critic(seed + 1000, input_size=input_size)
    actor_optimizer = Adam(actor.parameters(), learning_rate=0.001)
    critic_optimizer = Adam(critic.parameters(), learning_rate=0.001)
    recent_rewards = []

    for number in range(1, episodes + 1):
        reward, actor_loss, critic_loss = train_one_episode(
            actor, critic, game, actor_optimizer, critic_optimizer, discount, lam
        )
        recent_rewards.append(reward)
        if number % 100 == 0:
            print(
                "Episode",
                number,
                "recent reward:",
                round(sum(recent_rewards[-100:]) / 100, 2),
                "actor loss:",
                round(actor_loss, 3),
                "critic loss:",
                round(critic_loss, 3),
            )

    return actor, critic, actor_optimizer, critic_optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ghost", action="store_true")
    parser.add_argument("--local-view", action="store_true")
    args = parser.parse_args()
    if args.local_view and not args.ghost:
        parser.error("--local-view requires --ghost")

    actor, critic, actor_optimizer, critic_optimizer = train(
        episodes=args.episodes, seed=args.seed, lam=args.lam,
        with_ghost=args.ghost, full_view=not args.local_view
    )
    results = evaluate_policy(
        PelletMaze(with_ghost=args.ghost, full_view=not args.local_view),
        actor.choose_action, 100
    )
    print("Evaluation:", results)

    if args.local_view:
        prefix = "ghost_local_actor_critic"
    elif args.ghost:
        prefix = "ghost_actor_critic"
    else:
        prefix = "actor_critic"
    if args.lam != 1.0:
        prefix += "_lambda"
    folder = Path(__file__).parent / "checkpoints"
    folder.mkdir(exist_ok=True)
    save_checkpoint(folder / (prefix + "_actor.npz"), actor.network, actor_optimizer)
    save_checkpoint(folder / (prefix + "_critic.npz"), critic.network, critic_optimizer)
    print("Saved", folder / (prefix + "_actor.npz"), "and", folder / (prefix + "_critic.npz"))
