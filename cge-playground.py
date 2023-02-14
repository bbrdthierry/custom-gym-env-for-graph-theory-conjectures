import cge_custom_env as custom_env

# Test the interaction with environment by generating action sample from the environment.

env = custom_env.GraphEnv(N_vertices=30)

episodes = 100000

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()