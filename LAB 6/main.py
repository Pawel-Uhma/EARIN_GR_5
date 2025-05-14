import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


def choose_action(Q, state, eps, env):
    # pick an action: sometimes random, sometimes best guess
    if np.random.rand() < eps:
        return env.action_space.sample() 
    return int(np.argmax(Q[state]))      # pick best known move


def update_q(Q, state, action, reward, next_state, alpha, gamma):
    # core q-learning update step
    best_next = int(np.argmax(Q[next_state]))
    target = reward + gamma * Q[next_state, best_next]
    # move q[state, action] a bit towards target
    Q[state, action] += alpha * (target - Q[state, action])


def run_episode(env, Q, alpha, gamma, eps):
    # play one episode, track reward and steps
    state, _ = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        act = choose_action(Q, state, eps, env)  # decide what to do
        nxt, r, done, truncated, _ = env.step(act)
        total_r += r
        update_q(Q, state, act, r, nxt, alpha, gamma)  # learn from this move
        state = nxt
        steps += 1
    return total_r, steps  # return how well we did


def train(env, episodes, alpha, gamma, eps_start, eps_decay, eps_min, seed=None):
    # set random seed for repeatability
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)

    # init q-table with zeros
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q = np.zeros((n_s, n_a))

    eps = eps_start
    rewards = []
    lengths = []
    start = time.time()
    # breakpoints for prints
    check = set([1,2,5] + list(range(episodes//10, episodes+1, episodes//10)))

    for ep in range(1, episodes+1):
        r, l = run_episode(env, Q, alpha, gamma, eps)
        rewards.append(r)
        lengths.append(l)

        # decay epsilon so we explore less over time
        eps = max(eps_min, eps * eps_decay)

        # print progress at some eps
        if ep in check:
            elapsed = time.time() - start
            avg_r = np.mean(rewards[-(episodes//10 or 1):])
            avg_l = np.mean(lengths[-(episodes//10 or 1):])
            print(f"ep {ep}/{episodes} | avg r {avg_r:.2f} | avg len {avg_l:.1f} | eps {eps:.3f} | time {elapsed:.1f}s")

    print(f"finished training in {time.time()-start:.1f}s")
    return Q, rewards, lengths


def evaluate_policy(env, Q, trials=1000):
    # test out the greedy strategy and see how often it wins
    wins = 0
    lens = []
    for _ in range(trials):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done:
            a = int(np.argmax(Q[state]))  # always pick best
            state, r, done, truncated, _ = env.step(a)
            steps += 1
        wins += int(r)
        lens.append(steps)
    # report numbers
    win_rate = wins / trials
    print(f"evaluation: win_rate={win_rate*100:.1f}% | avg_len={np.mean(lens):.1f} | min/max={np.min(lens)}/{np.max(lens)}")
    return win_rate


def test_parameters(env, experiments, episodes=50000):
    # loop over settings and gather results for plots
    data = {}
    for exp in experiments:
        label = exp.get('label', '')
        print(f"\n-- testing [{label}] --")
        Q, rs, ls = train(
            env,
            episodes,
            exp.get('learning_rate', .8),
            exp.get('discount_factor', .99),
            exp.get('epsilon', 1.0),
            exp.get('epsilon_decay', .9995),
            exp.get('epsilon_min', .01),
            exp.get('seed', 42)
        )
        data[label] = (rs, ls)

    # plot success rates over episodes
    plt.figure()
    for k, (rs, _) in data.items():
        plt.plot(rs, label=k)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('rewards_plot.png')
    plt.close()

    # plot episode lengths over episodes
    plt.figure()
    for k, (_, ls) in data.items():
        plt.plot(ls, label=k)
    plt.xlabel('episode')
    plt.ylabel('length')
    plt.legend()
    plt.savefig('lengths_plot.png')
    plt.close()
    print('plots saved: rewards_plot.png, lengths_plot.png')


def main():
    # create the frozenlake env
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)
    Q, _, _ = train(env, episodes=50000, alpha=0.8, gamma=0.99, eps_start=1.0, eps_decay=0.9995, eps_min=0.01, seed=42)
    evaluate_policy(env, Q)

    experiments = [
            {"label": "base", "learning_rate": 0.8, "discount_factor": 0.99, "epsilon_decay": 0.9995},
            {"label": "low_lr", "learning_rate": 0.1, "discount_factor": 0.99, "epsilon_decay": 0.9995},
            {"label": "high_lr", "learning_rate": 1.0, "discount_factor": 0.99, "epsilon_decay": 0.9995},
            {"label": "fast_decay", "learning_rate": 0.8, "discount_factor": 0.99, "epsilon_decay": 0.995},
            {"label": "slow_decay", "learning_rate": 0.8, "discount_factor": 0.99, "epsilon_decay": 0.9999},
        ]
    test_parameters(env, experiments)

    env.close()


if __name__ == '__main__':
    main()
