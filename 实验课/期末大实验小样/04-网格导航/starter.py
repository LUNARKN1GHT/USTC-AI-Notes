"""Inspect expert trajectories and evaluate two simple navigation policies."""

import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import h5py
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fourrooms-expert-v0.hdf5"
SEED = 42
ACTION_NAMES = ("turn left", "turn right", "move forward")


def load_trajectories():
    episodes = []
    with h5py.File(DATA, "r") as source:
        names = sorted(source, key=lambda name: int(name.rsplit("_", 1)[1]))
        for name in names:
            episode = source[name]
            images = episode["observations/image"][:]
            actions = episode["actions"][:]
            rewards = episode["rewards"][:]
            assert len(images) == len(actions) + 1
            assert len(rewards) == len(actions)
            assert np.all((actions >= 0) & (actions <= 2))
            episodes.append(
                {
                    "name": name,
                    "seed": int(episode.attrs["seed"]),
                    "actions": actions,
                }
            )
        first = source[names[0]]
        print("First episode shapes:")
        print("  observations:", first["observations/image"].shape)
        print("  actions:", first["actions"].shape)
        print("  rewards:", first["rewards"].shape)
        print("  first actions:", first["actions"][:8].tolist())
        print("  episode return:", round(float(first["rewards"][:].sum()), 3))
    return episodes


def evaluate(env, seeds, policy):
    successes = 0
    returns = []
    lengths = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        episode_return = 0.0
        steps = 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            episode_return += reward
            steps += 1
            if terminated or truncated:
                successes += int(terminated and reward > 0)
                break
        returns.append(episode_return)
        lengths.append(steps)
    return successes / len(seeds), float(np.mean(returns)), float(np.mean(lengths))


def main():
    episodes = load_trajectories()
    order = np.random.default_rng(SEED).permutation(len(episodes))
    boundary = int(0.8 * len(order))
    train = [episodes[i] for i in order[:boundary]]
    test = [episodes[i] for i in order[boundary:]]
    train_actions = np.concatenate([episode["actions"] for episode in train])
    test_actions = np.concatenate([episode["actions"] for episode in test])
    counts = np.bincount(train_actions, minlength=3)
    majority = int(counts.argmax())
    print(f"Whole-episode split: {len(train)} train, {len(test)} test")
    print("Training action counts:", dict(zip(ACTION_NAMES, counts.tolist())))
    print("Most common action:", ACTION_NAMES[majority])
    print(f"Expert-action test accuracy: {(test_actions == majority).mean():.3f}")
    print("Object codes:", OBJECT_TO_IDX)

    # Compare both policies on exactly the same held-out starting scenarios.
    seeds = [episode["seed"] for episode in test[:20]]
    env = gym.make("MiniGrid-FourRooms-v0")
    rng = np.random.default_rng(SEED + 1)
    try:
        policies = {
            "random": lambda obs: int(rng.integers(3)),
            "most_common_action": lambda obs: majority,
        }
        for name, policy in policies.items():
            success, mean_return, mean_steps = evaluate(env, seeds, policy)
            print(
                f"{name}: {len(seeds)} episodes, success={success:.3f}, "
                f"return={mean_return:.3f}, steps={mean_steps:.1f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
