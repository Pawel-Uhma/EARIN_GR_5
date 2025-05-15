import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os       
import time


# global constants for our experiment settings
_BATCH = 100            # how many episodes to average in each data point
_EPISODES = 20000       # total episodes to run for each experiment
_TRIALS = 1000          # number of evaluation trials to run

def _ensure_dir(path: str) -> None:
    # make sure the directory exists, creating it if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _batched_average(values: list[float] | list[int]) -> list[float]:
    # takes a list of values and chunks them into batches of size _BATCH
    # returns the average of each batch - helps smooth out the noisy data
    return [
        float(np.mean(values[i:i + _BATCH]))
        for i in range(0, len(values), _BATCH)
    ]


def _plot_metric(values: list[float] | list[int],
                 ylabel: str,
                 title: str,
                 filename: str) -> None:
    # plots a single metric (like rewards or episode lengths)
    # this handles all the matplotlib setup, averaging, and saving
    y = _batched_average(values)
    x = [i * _BATCH for i in range(1, len(y) + 1)]   # episode number of batch end

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(f"episode (averaged over {_BATCH})")
    plt.ylabel(ylabel)
    plt.title(title)
    _ensure_dir(filename)
    plt.savefig(filename, dpi=120)
    plt.close()


def plot_rewards_single(label: str, rewards: list[float]) -> None:
    # convenient wrapper to plot just the rewards for one experiment
    fname = f"./LAB 6/plots/{label}_rewards_avg{_BATCH}.png"
    title = f"Average reward per {_BATCH} episodes – {label}"
    _plot_metric(rewards, "avg reward", title, fname)


def plot_lengths_single(label: str, lengths: list[int]) -> None:
    # convenient wrapper to plot just the episode lengths for one experiment
    fname = f"./LAB 6/plots/{label}_lengths_avg{_BATCH}.png"
    title = f"Average length per {_BATCH} episodes – {label}"
    _plot_metric(lengths, "avg length", title, fname)

def _plot_winrate(values: list[float], ylabel: str, title: str, filename: str) -> None:
    # computes the average win rate over the last 100 episodes
    y = [np.mean(values[i:i + 100]) for i in range(0, len(values) - 100 + 1)]
    x = [i + 100 for i in range(len(y))]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    _ensure_dir(filename)
    plt.savefig(filename, dpi=120)
    plt.close()


def plot_winrate_single(label: str, rewards: list[float]) -> None:
    # creates a plot of the win rate over the last 100 episodes
    fname = f"./LAB 6/plots/{label}_winrate_last100.png"
    title = f"Win rate over the last 100 episodes – {label}"
    _plot_winrate(rewards, "win rate (last 100)", title, fname)

# ------------- RL implementation ---------------------------------------------
def choose_action(Q, state, eps, env):
    # epsilon-greedy action selection - either explore randomly or exploit Q values
    # returns a random action with probability eps, otherwise returns best action
    explore = np.random.rand() < eps
    return env.action_space.sample() if explore else int(np.argmax(Q[state]))


def update_q(Q, state, action, reward, next_state, alpha, gamma):
    # implements the q-learning update rule to improve our value estimates
    # alpha controls how much we learn from each new experience
    # gamma is the discount factor - how much we care about future rewards
    prev       = Q[state, action]
    best_next  = int(np.argmax(Q[next_state]))
    target     = reward + gamma * Q[next_state, best_next]
    Q[state, action] += alpha * (target - prev)


def run_episode(env, Q, alpha, gamma, eps):
    # runs a single episode of interaction with the environment
    # keeps track of total reward and steps taken before termination
    state, _ = env.reset() #resets the board
    total_r, steps = 0.0, 0
    while True:
        act = choose_action(Q, state, eps, env) # epsilon-greedy action selection
        nxt, r, done, truncated, _ = env.step(act) # move
        update_q(Q, state, act, r, nxt, alpha, gamma) #update the Q function
        state = nxt
        total_r += r
        steps   += 1
        if done or truncated:
            break
    return total_r, steps


def train(env, episodes, alpha, gamma, eps_start, eps_decay, eps_min, seed=None):
    # main training loop for q-learning
    # handles epsilon decay, episode tracking, and logging progress
    print(f"Starting training | episodes={episodes}, α={alpha}, γ={gamma}, "
          f"ε₀={eps_start}, decay={eps_decay}, εmin={eps_min}, seed={seed}")
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)

    Q        = np.zeros((env.observation_space.n, env.action_space.n))
    eps      = eps_start
    rewards, lengths = [], []
    start    = time.time()
    # log progress more frequently at start and then at 10% intervals
    check    = set([1, 5, 10] + list(range(episodes // 10, episodes + 1, episodes // 10)))

    for ep in range(1, episodes + 1):
        r, l = run_episode(env, Q, alpha, gamma, eps)
        rewards.append(r)
        lengths.append(l)
        eps = max(eps_min, eps * eps_decay)  # decay epsilon but don't go below min

        if ep in check:
            elapsed = time.time() - start
            print(f"[train] ep {ep}/{episodes} | "
                  f"avg_r={np.mean(rewards[-len(rewards)//10 or -1:]):.3f} "
                  f"| avg_len={np.mean(lengths[-len(lengths)//10 or -1:]):.1f} "
                  f"| ε={eps:.3f} | {elapsed:.1f}s")

    print(f"Finished training in {time.time() - start:.1f}s | final ε={eps:.3f}")
    return Q, rewards, lengths


def evaluate_policy(env, Q, trials=_TRIALS):
    # test how good our trained policy is by running it without exploration
    # measures win rate and average episode length across multiple trials
    wins, lens = 0, []
    for _ in range(trials):
        state, _ = env.reset() #reset the board
        steps = 0
        while True:
            a = int(np.argmax(Q[state]))  # always pick best action (no exploration)
            state, r, done, truncated, _ = env.step(a)
            steps += 1
            if done or truncated:
                break
        wins += int(r)  # in frozen lake, reward of 1 means we reached the goal
        lens.append(steps)
    print(f"Win-rate {wins/trials*100:.1f}% | avg_len={np.mean(lens):.1f}")
    return wins / trials

def test_parameters(env, experiments, episodes=_EPISODES):
    # runs multiple experiments with different hyperparameters
    # good for comparing how different settings affect performance
    print(f"Running {len(experiments)} experiments, {episodes} episodes each")

    for exp in experiments:
        label = exp.get("label", "exp")
        print(f"\n-- {label} --")
        Q, rs, ls = train(
            env, episodes,
            alpha = exp.get("learning_rate", 0.8),
            gamma = exp.get("discount_factor", 0.99),
            eps_start=exp.get("epsilon",         1.0),
            eps_decay=exp.get("epsilon_decay",   0.9995),
            eps_min=exp.get("epsilon_min",     0.01),
            seed=exp.get("seed",            42)
        )
        evaluate_policy(env, Q)

        # save plots for this particular run
        plot_rewards_single(label, rs)
        plot_lengths_single(label, ls)
        plot_winrate_single(label, rs)


    print("Saved per-run plots to ./LAB6/")

def main():
    # creates the frozen lake environment and runs our experiments
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    Q, _, _ = train(env, episodes=_EPISODES, alpha=0.8, gamma=0.99,
                    eps_start=1.0, eps_decay=0.9999, eps_min=0.1, seed=42)
    evaluate_policy(env, Q)

    # setup different experiments to compare hyperparameter effects
    experiments = [
        {"label": "base", "learning_rate": .8,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9999},
        {"label": "slightly_low_lr", "learning_rate": .7,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9999},
        {"label": "slightly_high_lr", "learning_rate": .9, "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9999},
        {"label": "faster_decay",  "learning_rate": .8,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9995},
        {"label": "slower_decay",  "learning_rate": .8,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .99995},
        {"label": "higher_eps_min", "learning_rate": .8,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.2, "epsilon_decay": .9999},
        {"label": "lower_eps_min",  "learning_rate": .8,  "discount_factor": .99, "epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": .9999},
        {"label": "slightly_higher_gamma", "learning_rate": .8, "discount_factor": .995, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9999},
        {"label": "slightly_lower_gamma", "learning_rate": .8, "discount_factor": .98, "epsilon": 1.0, "epsilon_min": 0.1, "epsilon_decay": .9999},
    ]

    test_parameters(env, experiments)
    env.close()


if __name__ == "__main__":
    main()