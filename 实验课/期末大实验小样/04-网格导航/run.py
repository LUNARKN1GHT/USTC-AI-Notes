"""Compare expert imitation and online navigation in two MiniGrid settings."""

import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import h5py
import matplotlib
import minigrid  # Registers the MiniGrid environments with Gymnasium.
import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from torch import nn
from torch.nn.functional import one_hot

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SEED = 42
OBJECT_IDS = max(OBJECT_TO_IDX.values()) + 1
N_ACTIONS = 3  # left, right, forward


def load_episodes(path, env):
    """Validate that the stored seeds reproduce the observations and get the wall layouts."""
    episodes = []
    with h5py.File(path) as source:
        for key in sorted(source, key=lambda key: int(key.rsplit("_", 1)[1])):
            episode = source[key]
            seed = int(episode.attrs["seed"])
            actions = episode["actions"][:]
            images = episode["observations/image"][:-1]
            obs, _ = env.reset(seed=seed)
            assert np.array_equal(obs["image"], images[0]), (key, seed)
            assert obs["direction"] == episode["observations/direction"][0], (key, seed)
            assert np.all((0 <= actions) & (actions < N_ACTIONS)), key
            # Group by the walls only so differing starts/goals on a map stay together.
            layout = (env.unwrapped.grid.encode()[:, :, 0] == OBJECT_TO_IDX["wall"]).tobytes()
            episodes.append((seed, layout, images[..., 0], actions))
    return episodes


def split_by_layout(episodes):
    layouts = list(dict.fromkeys(layout for _, layout, _, _ in episodes))
    shuffled = np.random.default_rng(SEED).permutation(len(layouts))
    test_layouts = {layouts[i] for i in shuffled[: len(layouts) // 5]}
    return (
        [e for e in episodes if e[1] not in test_layouts],
        [e for e in episodes if e[1] in test_layouts],
    )


def object_channels(images):
    return one_hot(torch.as_tensor(images, dtype=torch.long), OBJECT_IDS).permute(
        0, 3, 1, 2
    ).float()


def expert_steps(episodes):
    images = object_channels(np.concatenate([e[2] for e in episodes]))
    actions = torch.as_tensor(np.concatenate([e[3] for e in episodes]), dtype=torch.long)
    return images, actions


def fit_action_model(model, images, actions):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(18):
        for idx in torch.randperm(len(actions), generator=generator).split(128):
            loss = loss_fn(model(images[idx]), actions[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def rollouts(env, seeds, policy):
    rewards, steps_to_goal = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        steps = 0
        episode_return = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            steps += 1
            episode_return += reward
            done = terminated or truncated
        rewards.append(episode_return)
        if terminated and reward > 0:
            steps_to_goal.append(steps)
    return {
        "successes": len(steps_to_goal),
        "episodes": len(rewards),
        "success_rate": len(steps_to_goal) / len(rewards),
        "mean_return": float(np.mean(rewards)),
        "mean_steps_if_success": float(np.mean(steps_to_goal)) if steps_to_goal else None,
    }


def summarize_episodes(episodes):
    return {
        "episodes": len(episodes),
        "layouts": len({e[1] for e in episodes}),
        "steps": sum(len(e[3]) for e in episodes),
    }


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=1)


def fourrooms(path):
    env = gym.make("MiniGrid-FourRooms-v0")
    try:
        train, test = split_by_layout(load_episodes(path, env))
        x_train, y_train = expert_steps(train)
        x_test, y_test = expert_steps(test)
        seeds = [e[0] for e in test]
        majority = int(y_train.bincount().argmax())
        results = {
            "Always forward": {
                "expert_action_accuracy": float((y_test == majority).float().mean()),
                **rollouts(env, seeds, lambda obs: majority),
            }
        }
        assert majority == 2, majority

        torch.manual_seed(SEED)
        mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(OBJECT_IDS * 49, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS)
        )
        torch.manual_seed(SEED)
        cnn = nn.Sequential(
            nn.Conv2d(OBJECT_IDS, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 49, 32),
            nn.ReLU(),
            nn.Linear(32, N_ACTIONS),
        )
        for name, model in (("MLP", mlp), ("CNN", cnn)):
            fit_action_model(model, x_train, y_train)
            accuracy = float((predict(model, x_test) == y_test).float().mean())
            result = rollouts(
                env, seeds, lambda obs: int(predict(model, object_channels(obs["image"][None, ..., 0])))
            )
            results[name] = {"expert_action_accuracy": accuracy, **result}

        return {"train": summarize_episodes(train), "test": summarize_episodes(test), "models": results}
    finally:
        env.close()


def empty_grid():
    env = gym.make("MiniGrid-Empty-Random-5x5-v0")
    try:
        # Use the full position and orientation in this introductory control task.
        q = np.zeros((5, 5, 4, N_ACTIONS))
        rng = np.random.default_rng(SEED)

        def state():
            x, y = env.unwrapped.agent_pos
            return int(x), int(y), int(env.unwrapped.agent_dir)

        training_episodes = 800
        for episode in range(training_episodes):
            env.reset(seed=10_000 + episode)
            epsilon = max(0.08, 0.8 * (1 - episode / 600))
            done = False
            while not done:
                old_state = state()
                if rng.random() < epsilon:
                    action = int(rng.integers(N_ACTIONS))
                else:
                    values = q[old_state]
                    action = int(rng.choice(np.flatnonzero(values == values.max())))
                _, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                target = reward if done else reward + 0.98 * q[state()].max()
                key = (*old_state, action)
                q[key] += 0.2 * (target - q[key])

        test_seeds = range(20_000, 20_200)
        random_eval = np.random.default_rng(SEED + 1)
        return {
            "training_episodes": training_episodes,
            "policies": {
                "Random": rollouts(env, test_seeds, lambda obs: int(random_eval.integers(N_ACTIONS))),
                "Q-learning": rollouts(env, test_seeds, lambda obs: int(q[state()].argmax())),
            },
        }
    finally:
        env.close()


def plot_results(results, path):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9), layout="constrained")
    models = results["fourrooms"]["models"]
    x = np.arange(len(models))
    w = 0.36
    for key, label, offset, color in [
        ("expert_action_accuracy", "Expert action accuracy", -w / 2, "#5177A7"),
        ("success_rate", "Goal success", w / 2, "#33A380"),
    ]:
        bars = ax.bar(x + offset, [100 * m[key] for m in models.values()], w, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=list(models), ylim=(0, 108), ylabel="Percent", title="FourRooms: imitation vs navigation")
    ax.legend(frameon=False, fontsize=8)
    policies = results["empty_grid"]["policies"]
    bars = ax2.bar(list(policies), [p["mean_return"] for p in policies.values()], color=["#ADB9C8", "#33A380"], width=0.55)
    ax2.bar_label(bars, fmt="%.3f", padding=4)
    ax2.set(ylim=(0, 1.08), ylabel="Mean return", title="Empty 5×5: online Q-learning")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    results = {
        "fourrooms": fourrooms(HERE / "data/fourrooms-expert-v0.hdf5"),
        "empty_grid": empty_grid(),
    }
    (HERE / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    plot_results(results, HERE / "result.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
