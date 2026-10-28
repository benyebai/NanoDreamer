"""Online training/evaluation of the PyTorch MinAtar transformer adaptation."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
from minatar import Environment
import numpy as np
import torch
from nanodreamer.agent import Agent
from nanodreamer.config import Config
from nanodreamer.replay import Replay


def frame(game):
    return game.state().transpose(2, 0, 1).copy()


def evaluate(agent, game_name, episodes=100, seed=930000, cap=200):
    results = {}
    # Evaluation must not change the training random stream.
    with torch.random.fork_rng():
        for mode in ("random", "agent"):
            game = Environment(game_name)
            game.seed(seed)
            actions = tuple(map(int, game.minimal_action_set()))
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            scores = []
            lengths = []
            for _ in range(episodes):
                game.reset()
                images = [frame(game)]
                previous_actions = [agent.actions]
                score = 0.0
                for step in range(cap):
                    if mode == "random":
                        action = int(rng.integers(len(actions)))
                    else:
                        action = agent.action(images, previous_actions, rng)
                    reward, done = game.act(actions[action])
                    score += float(reward)
                    images.append(frame(game))
                    previous_actions.append(action)
                    images = images[-agent.config.context :]
                    previous_actions = previous_actions[-agent.config.context :]
                    if done:
                        break
                scores.append(score)
                lengths.append(step + 1)
            results[mode] = dict(mean=float(np.mean(scores)), scores=scores, lengths=lengths)
    return dict(seed=seed, episodes=episodes, cap=cap, **results)


def save(folder, agent, replay, game, state, rng):
    folder = Path(folder)
    checkpoint = folder / "checkpoints" / str(state["step"])
    checkpoint.mkdir(parents=True, exist_ok=True)
    data = dict(
        config=asdict(agent.config),
        channels=frame(game).shape[0],
        actions=agent.actions,
        world=agent.world.state_dict(),
        encoder_target=agent.target_encoder.state_dict(),
        actor=agent.actor.state_dict(),
        critic=agent.critic.state_dict(),
        critic_target=agent.target_critic.state_dict(),
        optimizers=[
            optimizer.state_dict()
            for optimizer in (agent.world_opt, agent.actor_opt, agent.critic_opt)
        ],
        density=agent.density,
        return_scale=agent.return_scale,
        game=game,
        state=state,
        replay=replay,
        rng=rng,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    torch.save(data, checkpoint / "resume.pt")
    model = {
        name: value
        for (name, value) in data.items()
        if name not in ("game", "state", "replay", "rng", "torch_rng", "cuda_rng", "optimizers")
    }
    torch.save(model, checkpoint / "model.pt")
    (folder / "latest.tmp.json").write_text(
        json.dumps(dict(step=state["step"], checkpoint=str(checkpoint))) + "\n"
    )
    os.replace(folder / "latest.tmp.json", folder / "latest.json")


def load(folder, device):
    pointer = json.loads((Path(folder) / "latest.json").read_text())
    data = torch.load(
        Path(pointer["checkpoint"]) / "resume.pt", map_location=device, weights_only=False
    )
    agent = Agent(data["channels"], data["actions"], device, Config(**data["config"]))
    for name, key in (
        ("world", "world"),
        ("target_encoder", "encoder_target"),
        ("actor", "actor"),
        ("critic", "critic"),
        ("target_critic", "critic_target"),
    ):
        getattr(agent, name).load_state_dict(data[key])
    for optimizer, saved_state in zip(
        (agent.world_opt, agent.actor_opt, agent.critic_opt), data["optimizers"]
    ):
        optimizer.load_state_dict(saved_state)
    agent.density = data["density"].to(device)
    agent.return_scale = data["return_scale"]
    torch.set_rng_state(data["torch_rng"].cpu())
    if data["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in data["cuda_rng"]])
    return (agent, data["replay"], data["game"], data["state"], data["rng"])


def run(
    game_name,
    folder,
    steps=500000,
    max_seconds=12600,
    seed=0,
    resume=False,
    final_episodes=100,
    final_cap=1000,
):
    folder = Path(folder)
    if steps <= 0 or max_seconds <= 0 or final_episodes <= 0 or final_cap <= 0:
        raise ValueError("Steps, time limit, and evaluation episodes must be positive.")
    if resume and (not (folder / "latest.json").is_file()):
        raise FileNotFoundError(f"Cannot resume without checkpoint: {folder}")
    if not resume and folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Fresh training requires an empty output folder: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if resume and (folder / "latest.json").exists():
        agent, replay, game, state, rng = load(folder, device)
    else:
        game = Environment(game_name)
        game.seed(seed)
        game.reset()
        actions = tuple(map(int, game.minimal_action_set()))
        config = Config()
        agent = Agent(frame(game).shape[0], len(actions), device, config)
        replay = Replay(config, frame(game).shape[0], len(actions), seed)
        rng = np.random.default_rng(seed)
        replay.start(frame(game))
        for _ in range(config.seed_steps):
            action = int(rng.integers(len(actions)))
            reward, done = game.act(actions[action])
            replay.add(action, float(reward), float(not done), frame(game))
            if done:
                game.reset()
                replay.start(frame(game))
        game.reset()
        replay.start(frame(game))
        state = dict(
            step=0,
            images=[frame(game)],
            previous=[len(actions)],
            score=0.0,
            episode_steps=0,
            recent_scores=[],
            updates=0,
        )
    actions = tuple(map(int, game.minimal_action_set()))
    config = agent.config
    parameters = sum(
        (
            parameter.numel()
            for network in (agent.world, agent.actor, agent.critic)
            for parameter in network.parameters()
        )
    )
    metadata = dict(
        game=game_name,
        seed=seed,
        device=device,
        parameters=parameters,
        config=asdict(config),
        target_steps=steps,
    )
    (folder / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(event="start", step=state["step"], **metadata)), flush=True)
    start_time = last_log_time = last_save_time = time.monotonic()
    starting_step = state["step"]
    metrics = {}
    update_seconds = 0.0
    while state["step"] < steps and time.monotonic() - start_time < max_seconds:
        epsilon = max(0.05, 0.4 * (1 - state["step"] / 30000))
        action = agent.action(state["images"], state["previous"], rng, epsilon)
        reward, done = game.act(actions[action])
        observation = frame(game)
        replay.add(action, float(reward), float(not done), observation)
        state["step"] += 1
        state["episode_steps"] += 1
        state["score"] += float(reward)
        state["images"].append(observation)
        state["previous"].append(action)
        state["images"] = state["images"][-config.context :]
        state["previous"] = state["previous"][-config.context :]
        if done or state["episode_steps"] >= 1000:
            with (folder / "episodes.jsonl").open("a") as log_file:
                log_file.write(
                    json.dumps(
                        dict(
                            step=state["step"],
                            score=state["score"],
                            length=state["episode_steps"],
                            terminal=bool(done),
                        )
                    )
                    + "\n"
                )
            state["recent_scores"].append(state["score"])
            state["recent_scores"] = state["recent_scores"][-100:]
            game.reset()
            replay.start(frame(game))
            state.update(images=[frame(game)], previous=[agent.actions], score=0.0, episode_steps=0)
        if state["step"] % config.train_every == 0:
            update_start = time.monotonic()
            metrics = agent.update(replay)
            if device == "cuda":
                torch.cuda.synchronize()
            update_seconds += time.monotonic() - update_start
            state["updates"] += 1
            if not all((np.isfinite(metric) for metric in metrics.values())):
                raise RuntimeError(metrics)
        now = time.monotonic()
        if now - last_log_time >= 30:
            row = dict(
                event="progress",
                step=state["step"],
                seconds=now - start_time,
                steps_per_second=(state["step"] - starting_step) / (now - start_time),
                updates=state["updates"],
                recent_training_score=(
                    float(np.mean(state["recent_scores"])) if state["recent_scores"] else None
                ),
                **metrics,
            )
            with (folder / "metrics.jsonl").open("a") as log_file:
                log_file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            last_log_time = now
        if state["step"] % 50000 == 0:
            evaluated = evaluate(agent, game_name, episodes=100, seed=920000 + state["step"])
            (folder / f"eval_{state['step']}.json").write_text(
                json.dumps(evaluated, indent=2) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        event="evaluation",
                        step=state["step"],
                        agent=evaluated["agent"]["mean"],
                        random=evaluated["random"]["mean"],
                    )
                ),
                flush=True,
            )
            save(folder, agent, replay, game, state, rng)
            last_save_time = time.monotonic()
        elif now - last_save_time >= 300:
            save(folder, agent, replay, game, state, rng)
            last_save_time = time.monotonic()
    elapsed = time.monotonic() - start_time
    save(folder, agent, replay, game, state, rng)
    evaluated = evaluate(agent, game_name, episodes=final_episodes, seed=950000, cap=final_cap)
    (folder / "fresh_eval.json").write_text(json.dumps(evaluated, indent=2) + "\n")
    result = dict(
        **metadata,
        final_step=state["step"],
        elapsed_seconds=elapsed,
        steps_per_second=(state["step"] - starting_step) / elapsed,
        training_seconds=update_seconds,
        evaluation_episodes=final_episodes,
        evaluation_cap=final_cap,
        agent_mean=evaluated["agent"]["mean"],
        random_mean=evaluated["random"]["mean"],
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="completed", **result)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--seconds", type=int, default=12600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-cap", type=int, default=1000)
    args = parser.parse_args()
    run(
        args.game,
        args.folder,
        args.steps,
        args.seconds,
        args.seed,
        args.resume,
        final_episodes=args.eval_episodes,
        final_cap=args.eval_cap,
    )
