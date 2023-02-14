import cge_custom_env as custom_env
from gym.wrappers import FlattenObservation
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, A2C, ACKTR # DQN : Deep Q Learning, A2C : Actor Critic, ACKTR : Actor-Critic using Kronecker-Factored Trust Region
# more info about RL algorithms here : https://stable-baselines.readthedocs.io/en/master/guide/algos.html

# Resize observations, otherwise RL algorithms won't work
env = FlattenObservation(custom_env.GraphEnv(N_vertices=17))

# Test with ACKTR algorithm. If using DQN, must uncomment "from stable_baselines.deepq.policies import MlpPolicy" above
model = ACKTR(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    print("action et states:")
    print(action, _states)
    obs, rewards, dones, info = env.step(action)
    print("rewards:", rewards)
