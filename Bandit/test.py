import numpy as np

# state = {L, H}
L, H = 0, 1

# action_set = {search, wait, recharge}
search, wait, recharge = 1, 2, 3
action_set_H = [search, wait]
action_set_L = [search, wait, recharge]

# Hyperparameters: reward and prob.
alpha, beta = 0.5, 0.9  # The prob of from H to L, and from L to H (rescued), respectively.
r_wait, r_search = 0, 1


def take_action(state, action):
    if state == H:
        if action == wait:
            p = 1.
            reward = np.array([r_wait, r_wait])
        elif action == search:
            p = alpha
            reward = np.array([r_search, r_search])
        else:
            raise Exception("Invalid action at state H. Note that there is no action 'recharge' at all.")

    elif state == L:
        if action == wait:
            p = 1.
            reward = np.array([r_wait, r_wait])
        elif action == search:
            p = beta
            reward = np.array([r_search, -3])
        elif action == recharge:
            p = 0.
            reward = np.array([0, 0])
        else:
            raise Exception("Invalid action.")

    else:
        raise Exception("Invalid state.")

    return np.array([p, 1 - p]), reward


# Init the state value function
V = np.array([0., 0.])

gamma = 1.  # Hyperparameters: tolerance
for k in range(100):  # For each iteration
    # Evaluate the state H.
    for action in action_set_H:
        p, r = take_action(H, action)
        V[H] += 0.5 * np.sum((p * (r + gamma * np.array(V))))

    for action in action_set_L:
        p, r = take_action(L, action)
        V[L] += 0.5 * np.sum((p * (r + gamma * np.array(V))))

    print('Iteration %d, V: %s' % (k + 1, str(V)))
