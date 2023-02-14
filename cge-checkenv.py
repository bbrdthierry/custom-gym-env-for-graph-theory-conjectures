import cge_custom_env as custom_env
from gym.wrappers import FlattenObservation
from stable_baselines.common.env_checker import check_env

# Check if the custom gym environment is compatible with stable-baseline API

env = custom_env.GraphEnv(N_vertices=17)

wrapped_env = FlattenObservation(env)
reset_obs = wrapped_env.reset()

print("Shapes:")
print("reset_obs shape:", reset_obs.shape, "type:", type(reset_obs))
print("normal_obs shape:", wrapped_env.observation_space.sample().shape, "type:", type(wrapped_env.observation_space.sample()))

check_env(env, warn=True)