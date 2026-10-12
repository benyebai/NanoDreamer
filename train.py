from pathlib import Path

from mininn.checkpoint import save_checkpoint
from mininn.optimizers import Adam, clip_grad_norm
from tensorgrad import Tensor

from tinydreamer.actor import Actor
from tinydreamer.evaluate import evaluate_policy
from tinydreamer.game import PelletMaze
from tinydreamer.returns import calculate_returns
from tinydreamer.trajectory import collect_episode


def train_one_episode(actor, game, optimizer, discount):
    episode = collect_episode(game, actor.choose_action)
    returns = calculate_returns(episode, discount)

    # REINFORCE policy loss
    loss = Tensor(0.0)
    for step, future_reward in zip(episode, returns):
        scores = actor.forward(step["observation"])
        # probabilities = scores.softmax(axis=1)
        # chosen_log_prob = probabilities[0, step["action"]].log()
        # same idea as below
        log_probs = scores - scores.logsumexp(axis=1, keepdims=True)
        # get the log probability of the actual action chosen
        chosen_log_prob = log_probs[0, step["action"]]
        loss = loss - chosen_log_prob * future_reward

    loss = loss / len(episode)
    actor.network.zero_grad()
    loss.backward()
    clip_grad_norm(actor.parameters(), 5.0)
    optimizer.step()

    total_reward = sum(step["reward"] for step in episode)
    return total_reward, float(loss.data)


def train(episodes=500, seed=0, discount=0.99):
    actor = Actor(seed)
    game = PelletMaze()
    optimizer = Adam(actor.parameters(), learning_rate=0.001)
    recent_rewards = []

    for number in range(1, episodes + 1):
        reward, loss = train_one_episode(actor, game, optimizer, discount)
        recent_rewards.append(reward)
        if number % 100 == 0:
            print(
                "Episode",
                number,
                "average reward over last 100:",
                round(sum(recent_rewards[-100:]) / 100, 2),
                "loss:",
                round(loss, 3),
            )

    return actor, optimizer


if __name__ == "__main__":
    actor, optimizer = train()
    results = evaluate_policy(PelletMaze(), actor.choose_action, 100)
    print("Evaluation after training:", results)
    checkpoint = Path(__file__).parent / "checkpoints" / "actor_checkpoint.npz"
    checkpoint.parent.mkdir(exist_ok=True)
    save_checkpoint(checkpoint, actor.network, optimizer)
    print("Saved", checkpoint)
