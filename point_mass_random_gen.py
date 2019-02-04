from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np


class RandPolicy:
    name = "random"
    discrete_action = False

    def __init__(self, observation_space, action_space, num_envs):
        print(type(action_space))
        if isinstance(action_space, Box):
            self.act_size = len(action_space.shape)
            self.act_low = action_space.low
            self.act_high = action_space.high
        elif isinstance(action_space, Discrete):
            self.act_size = action_space.n
            self.act_low = 0
            self.act_high = self.act_size
            self.discrete_action = True

        self.num_envs = num_envs

    def act(self, obs):
        if self.discrete_action:
            return np.random.randint(self.act_high, size=[self.num_envs])
        else:
            return np.random.uniform(np.tile(self.act_low[None, ...], [self.num_envs, 1]),
                                     np.tile(self.act_high[None, ...], [self.num_envs, 1]))


def dynamics_data_gen(env_name='Reacher-v2', start_seed=0, timesteps=10, n_parallel_envs=1, width=300, height=240):
    import gym  # import locally so that caller can patch gym

    def make_env(seed):
        def _():
            env = gym.make(env_name)
            env.seed(seed)
            return env

        return _

    # Uncomment this to show the bug
    # from requests_futures.sessions import FuturesSession
    # session = FuturesSession()
    # session.get('http://www.google.com', )

    from subproc_vec_env import SubprocVecEnv
    # from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    env = SubprocVecEnv([make_env(s) for s in range(start_seed, start_seed + n_parallel_envs)])

    policy = RandPolicy(env.observation_space, env.action_space, env.num_envs)

    rollouts = []
    obs = env.reset()
    for i in range(timesteps):
        # fs = env.render("rgb", width=width, height=height)
        fs = env.render("rgb_array", width=width, height=height)
        acs = policy.act(obs)
        rollouts.append(dict(obs=obs, acs=acs, views=fs))
        obs, rewards, dones, infos = env.step(acs)

    import pandas as pd
    return {k: np.stack(v) for k, v in pd.DataFrame(rollouts).items()}


def main(env_id="Reacher-v2"):
    # env_name = "PointMass-v0"
    samples = dynamics_data_gen(env_name=env_id, start_seed=0, timesteps=50, n_parallel_envs=5, width=28, height=28)

    for k, v in samples.items():
        print(k, v.shape)

    video = np.swapaxes(samples['views'], 0, 1).reshape(-1, 28, 28, 3)
    from ml_logger import logger
    logger.log_video(video, 'test.mp4')


if __name__ == "__main__":
    from ge_world import IS_PATCHED

    assert IS_PATCHED, "need to patch"

    main("GoalMassDiscrete-v0")
