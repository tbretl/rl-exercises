import numpy as np
import matplotlib.pyplot as plt
import gridworld
from scipy.ndimage import uniform_filter1d


def policy_as_string(policy):
    output = ''
    k = 0
    for i in range(5):
        for j in range(5):
            s = i*5 + j
            a = policy[s]
            c = ['>','^','<','v'][a]
            output += c
            k += 1
        output += '\n'
    return output

def rollout(env, policy):
    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = policy[s]
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    return data


def value_iteration(rg):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Define hyperparameters
    gamma = 0.95
    delta_stop = 0.001

    # Train
    # TODO - Initialize the value function as a list (or 1D ndarray) of floats with
    #        length equal to the number of states (this information is available to
    #        you in the "env" object). It is common to sample initial values from a
    #        normal distribution.
    # V = ...
    # TODO - Initialize the policy as a list (or 1D ndarray) of integers with length
    #        equal to the number of states. Make sure these integers are in the range
    #        of the number of actions. It is common to sample initial actions from a
    #        uniform distribution.
    # policy = ...
    iter = 0
    training_data = {
        'iter': [],
        'V_mean': [],
        'delta': [],
    }
    while True:
        iter += 1
        # TODO - Initialize delta.
        # delta = ...
        for s in range(env.num_states):
            # TODO - Implement one value iteration:
            #
            # v = ...           # <-- float
            # Q = [ ... ]       # <-- list of floats
            # policy[s] = ...   # <-- integer
            # V[s] = ...        # <-- float
            # delta = ...       # <-- float
        training_data['iter'].append(iter)
        training_data['V_mean'].append(np.mean(V))
        training_data['delta'].append(delta)
        # TODO - Implement stopping criterion (replace "True" with a condition):
        if True:
            break

    # Plot mean of value function (i.e., learning curve)
    plt.figure()
    plt.plot(training_data['iter'], training_data['V_mean'])
    plt.grid()
    plt.xlabel('iterations')
    plt.legend(['mean of value function'])
    plt.savefig('gridworld_vi_learningcurve.png')

    # Plot max change in value function after each iteration (i.e., convergence)
    plt.figure()
    plt.semilogy(training_data['iter'], training_data['delta'])
    plt.grid()
    plt.xlabel('iterations')
    plt.legend(['maximum change in value function after each iteration'])
    plt.savefig('gridworld_vi_convergence.png')

    # Visualize rollout
    rollout_data = rollout(env, policy)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(rollout_data['t'], rollout_data['s'], '.-')
    ax[0].set_yticks([0, 5, 10, 15, 20])
    ax[0].set_ylim(0, 24)
    ax[0].set_ylabel('state')
    ax[0].grid()
    ax[1].plot(rollout_data['t'][:-1], rollout_data['a'], '.-')
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel('action')
    ax[1].grid()
    ax[2].plot(rollout_data['t'][:-1], rollout_data['r'], '.-')
    ax[2].set_ylim(-1, 10)
    ax[2].set_ylabel('reward')
    ax[2].grid()
    plt.tight_layout()
    plt.savefig('gridworld_vi_rollout.png')

    # Visualize policy
    print('\nVI: POLICY\n')
    print(policy_as_string(policy))

    # Visualize value function
    plt.figure()
    Vij = np.reshape(V, (5, 5))
    plt.pcolor(Vij)
    plt.ylim(5, 0)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('V(s)')
    plt.savefig('gridworld_vi_valuefunction.png')

    # TODO - Check if the value function is what we expect. In particular, compute
    # the total discounted reward that you would get if from starting in state 1. We
    # will compare this to V[1].
    # V_1 = ...
    print(f'V[1] = {V[1]:7.3f}, V_1 = {V_1:7.3f}')

def policy_iteration(rg):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Define hyperparameters
    gamma = 0.95
    delta_stop = 0.001

    # Train
    # TODO - Initialize the value function as a list (or 1D ndarray) of floats with
    #        length equal to the number of states (this information is available to
    #        you in the "env" object). It is common to sample initial values from a
    #        normal distribution.
    # V = ...
    # TODO - Initialize the policy as a list (or 1D ndarray) of integers with length
    #        equal to the number of states. Make sure these integers are in the range
    #        of the number of actions. It is common to sample initial actions from a
    #        uniform distribution.
    # policy = ...
    policy_iter = 0
    training_data = {
        'policy_iter': [],
        'V_iter': [],
        'V_mean': [],
    }
    done = False
    while not done:
        policy_iter += 1
        # Policy evaluation
        V_iter = 0
        while True:
            V_iter += 1
            # TODO - Initialize delta.
            # delta = ...
            for s in range(env.num_states):
                # TODO - Implement one value iteration:
                #
                # v = ...           # <-- float (value at current state)
                # a = ...           # <-- integer (action at current state)
                # V[s] = ...        # <-- float
                # delta = ...       # <-- float
            # TODO - Implement stopping criterion (replace "True" with a condition):
            if True:
                break
        # Policy improvement
        done = True
        for s in range(env.num_states):
            # TODO - Implement one policy iteration:
            #
            # a_old = ...           # <-- integer
            # Q = ...               # <-- list of floats
            # policy[s] = ...       # <-- integer
            #
            # TODO - Implement stopping criterion (replace "True" with a condition):
            if True:
                done = False
        print(f'{policy_iter} : {V_iter} : {np.mean(V):7.3f}')

        training_data['policy_iter'].append(policy_iter)
        training_data['V_iter'].append(V_iter)
        training_data['V_mean'].append(np.mean(V))

    # Plot mean of value function (i.e., learning curve)
    plt.figure()
    plt.plot(np.cumsum(training_data['V_iter']), training_data['V_mean'], '.-')
    plt.grid()
    plt.xlim(0, 50 * np.ceil(np.sum(training_data['V_iter']) / 50))
    plt.xlabel('value iterations')
    plt.legend(['mean of value function'])
    plt.savefig('gridworld_pi_learningcurve.png')

    # Visualize rollout
    rollout_data = rollout(env, policy)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(rollout_data['t'], rollout_data['s'], '.-')
    ax[0].set_yticks([0, 5, 10, 15, 20])
    ax[0].set_ylim(0, 24)
    ax[0].set_ylabel('state')
    ax[0].grid()
    ax[1].plot(rollout_data['t'][:-1], rollout_data['a'], '.-')
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel('action')
    ax[1].grid()
    ax[2].plot(rollout_data['t'][:-1], rollout_data['r'], '.-')
    ax[2].set_ylim(-1, 10)
    ax[2].set_ylabel('reward')
    ax[2].grid()
    plt.tight_layout()
    plt.savefig('gridworld_pi_rollout.png')

    # Visualize policy
    print('\nPI: POLICY\n')
    print(policy_as_string(policy))

    # Visualize value function
    plt.figure()
    Vij = np.reshape(V, (5, 5))
    plt.pcolor(Vij)
    plt.ylim(5, 0)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('V(s)')
    plt.savefig('gridworld_pi_valuefunction.png')

    # TODO - Check if the value function is what we expect. In particular, compute
    # the total discounted reward that you would get if from starting in state 1. We
    # will compare this to V[1].
    # V_1 = ...
    print(f'V[1] = {V[1]:7.3f}, V_1 = {V_1:7.3f}')


def train_sarsa(env, rg, gamma=0.95, alpha=0.5, epsilon=0.1, num_steps=1000000):
    print(f'\nTRAIN SARSA (gamma={gamma}, alpha={alpha}, epsilon={epsilon}, num_steps={num_steps})')
    # TODO - Initialize the Q-function as a 2D ndarray of floats with a number of
    #        rows equal to the number of states and a number of columns equal to the
    #        number of actions. It is common to sample initial Q-values from a
    #        normal distribution.
    # Q = ...
    step = 0
    training_data = {
        'step': [],
        'r_undiscounted': [],
        'r_discounted': [],
    }
    while step < num_steps:
        # TODO - Get initial state
        # s = ...
        # TODO - Get initial action
        # a = ...
        data = {
            's': [s],
            'a': [a],
            'r': [],
        }
        done = False
        while not done:
            step += 1
            # TODO - Implement one iteration of Sarsa:
            #
            # (s1, r, done) = ... # <-- step simulation (integer, float, boolean)
            # a1 = ...            # <-- choose action (integer)
            # Q[s, a] = ...       # <-- update Q-value (float)
            # s = ...             # <-- update state (integer)
            # a = ...             # <-- update action (integer)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)
        training_data['step'].append(step)
        training_data['r_undiscounted'].append(np.sum(data['r']))
        training_data['r_discounted'].append(np.sum(np.power(gamma, range(len(data['r']))) * np.array(data['r'])))
        if step % 1e4 == 0:
            print(f'{step:10d} : {training_data["r_undiscounted"][-1]:7.3f} : {training_data["r_discounted"][-1]:7.3f}')
    # TODO - Get the final policy (a list of integers with length equal to the number of actions)
    # policy = ...
    print('\n')
    return training_data, policy, Q


def sarsa(rg):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    gamma = 0.95

    # Place to store results
    results = []

    # Train for one set of hyperparameters
    alpha = 0.1
    epsilon = 0.1
    training_data, policy, Q = train_sarsa(env, rg, alpha=alpha, epsilon=epsilon)
    results.append({
        'step': training_data['step'],
        'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
        'alpha': alpha,
        'epsilon': epsilon,
    })

    # Estimate the value function corresponding to the learned policy with TD(0)
    td0_data, V = TD0(rg, env, policy)

    # Plot learning curve
    plt.figure()
    plt.plot(training_data['step'], training_data['r_discounted'], '.')
    plt.plot(training_data['step'], uniform_filter1d(training_data['r_discounted'], size=100), '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend(['total discounted reward', 'smoothed'])
    plt.savefig('gridworld_sarsa_learningcurve.png')

    # Plot learning curve for TD(0)
    plt.figure()
    plt.plot(td0_data['step'], td0_data['V_mean'], '.-')
    plt.grid()
    plt.xlabel('steps')
    plt.legend(['mean of value function'])
    plt.savefig('gridworld_sarsa_learningcurve_td0.png')

    # Visualize rollout
    rollout_data = rollout(env, policy)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(rollout_data['t'], rollout_data['s'], '.-')
    ax[0].set_yticks([0, 5, 10, 15, 20])
    ax[0].set_ylim(0, 24)
    ax[0].set_ylabel('state')
    ax[0].grid()
    ax[1].plot(rollout_data['t'][:-1], rollout_data['a'], '.-')
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel('action')
    ax[1].grid()
    ax[2].plot(rollout_data['t'][:-1], rollout_data['r'], '.-')
    ax[2].set_ylim(-1, 10)
    ax[2].set_ylabel('reward')
    ax[2].grid()
    plt.tight_layout()
    plt.savefig('gridworld_sarsa_rollout.png')

    # Visualize policy
    print('\nSARSA: POLICY\n')
    print(policy_as_string(policy))

    # Visualize value function
    plt.figure()
    Vij = np.reshape(V, (5, 5))
    plt.pcolor(Vij)
    plt.ylim(5, 0)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('V(s)')
    plt.savefig('gridworld_sarsa_valuefunction.png')

    # TODO - Check if the value function is what we expect. In particular, compute
    # the total discounted reward that you would get if from starting in state 1. We
    # will compare this to V[1].
    # V_1 = ...
    print(f'V[1] = {V[1]:7.3f}, V_1 = {V_1:7.3f}')

    # Train for different values of alpha
    epsilon = 0.1
    for alpha in [0.05, 0.2]:
        training_data, policy, Q = train_sarsa(env, rg, alpha=alpha, epsilon=epsilon)
        results.append({
            'step': training_data['step'],
            'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
            'alpha': alpha,
            'epsilon': epsilon,
        })

    # Train for different values of epsilon
    alpha = 0.1
    for epsilon in [0.05, 0.2]:
        training_data, policy, Q = train_sarsa(env, rg, alpha=alpha, epsilon=epsilon)
        results.append({
            'step': training_data['step'],
            'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
            'alpha': alpha,
            'epsilon': epsilon,
        })

    # Plot learning curves as alpha varies
    results.sort(key=lambda result: result['alpha'])
    plt.figure()
    alphas = []
    for res in results:
        if res['epsilon'] == 0.1:
            alphas.append(res['alpha'])
            plt.plot(res['step'], res['r_smoothed'], '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend([f'$\\alpha = {alpha}$' for alpha in alphas])
    plt.savefig('gridworld_sarsa_learningcurves_alpha.png')

    # Plot learning curves as epsilon varies
    results.sort(key=lambda result: result['epsilon'])
    plt.figure()
    epsilons = []
    for res in results:
        if res['alpha'] == 0.1:
            epsilons.append(res['epsilon'])
            plt.plot(res['step'], res['r_smoothed'], '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend([f'$\\epsilon = {epsilon}$' for epsilon in epsilons])
    plt.savefig('gridworld_sarsa_learningcurves_epsilon.png')


def train_q_learning(env, rg, gamma=0.95, alpha=0.5, epsilon=0.1, num_steps=1000000):
    print(f'\nTRAIN Q-LEARNING (gamma={gamma}, alpha={alpha}, epsilon={epsilon}, num_steps={num_steps})')
    # TODO - Initialize the Q-function as a 2D ndarray of floats with a number of
    #        rows equal to the number of states and a number of columns equal to the
    #        number of actions. It is common to sample initial Q-values from a
    #        normal distribution.
    # Q = ...
    step = 0
    training_data = {
        'step': [],
        'r_undiscounted': [],
        'r_discounted': [],
    }
    while step < num_steps:
        # TODO - Get initial state
        # s = ...
        data = {
            's': [s],
            'a': [],
            'r': [],
        }
        done = False
        while not done:
            step += 1
            # TODO - Implement one iteration of Q-learning:
            #
            # a = ...               # <-- choose action (integer)
            # (s1, r, done) = ...   # <-- step simulation (integer, float, boolean)
            # Q[s, a] = ...         # <-- update Q-value (float)
            # s = ...               # <-- update state (integer)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)
        training_data['step'].append(step)
        training_data['r_undiscounted'].append(np.sum(data['r']))
        training_data['r_discounted'].append(np.sum(np.power(gamma, range(len(data['r']))) * np.array(data['r'])))
        if step % 1e4 == 0:
            print(f'{step:10d} : {training_data["r_undiscounted"][-1]:7.3f} : {training_data["r_discounted"][-1]:7.3f}')
    # TODO - Get the final policy (a list of integers with length equal to the number of actions)
    # policy = ...
    print('\n')
    return training_data, policy, Q


def q_learning(rg):
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    gamma = 0.95

    # Place to store results
    results = []

    # Train for one set of hyperparameters
    alpha = 0.1
    epsilon = 0.1
    training_data, policy, Q = train_q_learning(env, rg, alpha=alpha, epsilon=epsilon)
    results.append({
        'step': training_data['step'],
        'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
        'alpha': alpha,
        'epsilon': epsilon,
    })

    # Estimate the value function corresponding to the learned policy with TD(0)
    td0_data, V = TD0(rg, env, policy)

    # Plot learning curve
    plt.figure()
    plt.plot(training_data['step'], training_data['r_discounted'], '.')
    plt.plot(training_data['step'], uniform_filter1d(training_data['r_discounted'], size=100), '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend(['total discounted reward', 'smoothed'])
    plt.savefig('gridworld_qlearning_learningcurve.png')

    # Plot learning curve for TD(0)
    plt.figure()
    plt.plot(td0_data['step'], td0_data['V_mean'], '.-')
    plt.grid()
    plt.xlabel('steps')
    plt.legend(['mean of value function'])
    plt.savefig('gridworld_qlearning_learningcurve_td0.png')

    # Visualize rollout
    rollout_data = rollout(env, policy)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(rollout_data['t'], rollout_data['s'], '.-')
    ax[0].set_yticks([0, 5, 10, 15, 20])
    ax[0].set_ylim(0, 24)
    ax[0].set_ylabel('state')
    ax[0].grid()
    ax[1].plot(rollout_data['t'][:-1], rollout_data['a'], '.-')
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel('action')
    ax[1].grid()
    ax[2].plot(rollout_data['t'][:-1], rollout_data['r'], '.-')
    ax[2].set_ylim(-1, 10)
    ax[2].set_ylabel('reward')
    ax[2].grid()
    plt.tight_layout()
    plt.savefig('gridworld_qlearning_rollout.png')

    # Visualize policy
    print('\nQ-LEARNING: POLICY\n')
    print(policy_as_string(policy))

    # Visualize value function
    plt.figure()
    Vij = np.reshape(V, (5, 5))
    plt.pcolor(Vij)
    plt.ylim(5, 0)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('V(s)')
    plt.savefig('gridworld_qlearning_valuefunction.png')

    # TODO - Check if the value function is what we expect. In particular, compute
    # the total discounted reward that you would get if from starting in state 1. We
    # will compare this to V[1].
    # V_1 = ...
    print(f'V[1] = {V[1]:7.3f}, V_1 = {V_1:7.3f}')

    # Train for different values of alpha
    epsilon = 0.1
    for alpha in [0.05, 0.2]:
        training_data, policy, Q = train_q_learning(env, rg, alpha=alpha, epsilon=epsilon)
        results.append({
            'step': training_data['step'],
            'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
            'alpha': alpha,
            'epsilon': epsilon,
        })

    # Train for different values of epsilon
    alpha = 0.1
    for epsilon in [0.05, 0.2]:
        training_data, policy, Q = train_q_learning(env, rg, alpha=alpha, epsilon=epsilon)
        results.append({
            'step': training_data['step'],
            'r_smoothed': uniform_filter1d(training_data['r_discounted'], size=100),
            'alpha': alpha,
            'epsilon': epsilon,
        })

    # Plot learning curves as alpha varies
    results.sort(key=lambda result: result['alpha'])
    plt.figure()
    alphas = []
    for res in results:
        if res['epsilon'] == 0.1:
            alphas.append(res['alpha'])
            plt.plot(res['step'], res['r_smoothed'], '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend([f'$\\alpha = {alpha}$' for alpha in alphas])
    plt.savefig('gridworld_qlearning_learningcurves_alpha.png')

    # Plot learning curves as epsilon varies
    results.sort(key=lambda result: result['epsilon'])
    plt.figure()
    epsilons = []
    for res in results:
        if res['alpha'] == 0.1:
            epsilons.append(res['epsilon'])
            plt.plot(res['step'], res['r_smoothed'], '-', linewidth=3)
    plt.grid()
    plt.xlabel('steps')
    plt.legend([f'$\\epsilon = {epsilon}$' for epsilon in epsilons])
    plt.savefig('gridworld_qlearning_learningcurves_epsilon.png')



def TD0(rg, env, policy, gamma=0.95, alpha=0.1, num_steps=250000):
    print(f'\nAPPLY TD(0) TO ESTIMATE VALUE FUNCTION (alpha={alpha}, num_steps={num_steps})')
    # TODO - Initialize the value function as a list (or 1D ndarray) of floats with
    #        length equal to the number of states (this information is available to
    #        you in the "env" object). It is common to sample initial values from a
    #        normal distribution.
    # V = ...
    step = 0
    training_data = {
        'step': [],
        'V_mean': [],
    }
    while step < num_steps:
        # TODO - Get initial state
        # s = ...
        done = False
        while not done:
            step += 1
            # TODO - Implement one iteration of TD0
            # a = ...               # <-- choose action (integer)
            # (s1, r, done) = ...   # <-- step simulation (integer, float, boolean)
            # V[s] = ...            # <-- update value (float)
            s = s1
        training_data['step'].append(step)
        training_data['V_mean'].append(np.mean(V))
        if step % 1e4 == 0:
            print(f'{step:10d} : {training_data["V_mean"][-1]:7.3f}')
    print('\n')
    return training_data, V

def main():
    rg = np.random.default_rng()
    value_iteration(rg)
    policy_iteration(rg)
    sarsa(rg)
    q_learning(rg)


if __name__ == '__main__':
    main()
