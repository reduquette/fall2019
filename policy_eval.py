import numpy as np
import sys
import math
from gridworld import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        done = True  #assume this is the last iteration

        for state in range(env.nS):
            new_value = sum([
                policy[state][action] *
                (env.P[state][action][0][2] + discount_factor * sum([
                    next_state[0] * V[next_state[1]]
                    for next_state in env.P[state][action]
                ])) for action in range(env.nA)
            ])
            # if any state changes more than theta, continue for another iteration
            if abs(V[state] - new_value) > theta:
                done = False
            V[state] = new_value

        if done:
            break
    return np.array(V)


def policy_adjust(env,
                  policy,
                  v,
                  in_place=True,
                  policy_eval=policy_eval,
                  discount_factor=1):
    """Policy improvement algorithm. Improves the given policy with respect to the given state value array. Called in policy_improvement."""
    # option to either create new array or modify policy in place
    if in_place:
        new_policy = policy
    else:
        new_policy = np.zeros_like(policy)

    # initialize flag denoting if policy has changed
    stable = True

    # iterate over states, greedily choosing best actions
    for state in range(env.nS):
        # find the value of each action
        action_values = []
        for action in env.P[state]:
            p_sprime_r = [
                prob * (reward + discount_factor * v[next_state])
                for prob, next_state, reward, is_done in env.P[state][action]
            ]
            action_values.append(sum(p_sprime_r))
        action_values = np.array(action_values)

        # find optimal action(s)
        # can't use direct comparison (==) because of float imprecision
        opt_actions = np.isclose(action_values, np.amax(action_values))
        optimal_policy = opt_actions / np.sum(opt_actions)

        if not np.all(np.isclose(optimal_policy, policy[state])):
            stable = False

        # update policy to uniform distribution over optimal actions at each state
        # print("values:", action_values, "actions:", opt_actions,
        #       "bad actions:", suboptimal)

        new_policy[state] = optimal_policy
        # new_policy[state][suboptimal] = 0

    return new_policy, stable


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        v = policy_eval_fn(policy, env)
        policy, stable = policy_adjust(
            env,
            policy,
            v,
            in_place=True,
            policy_eval=policy_eval_fn,
            discount_factor=discount_factor)
        if stable:
            break

    return policy, policy_eval_fn(policy, env)


# long version of the list comprehension above
# for action in range(env.nA):
#     prob_action = policy[state][action]
#     print(prob_action)
#     reward = env.P[state][action][0][2]
#     print(reward)
#     transitions = [
#         next_state[0] * V[next_state[1]]
#         for next_state in env.P[state][action]
#     ]
#     print(transitions)
#     res = prob_action * (reward + sum(transitions))
#     print(res)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
print(env.P)
# print(v[1])
print(policy_adjust(env, random_policy, v, in_place=False))

print(policy_improvement(env))
