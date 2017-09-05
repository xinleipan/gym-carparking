from gym.envs.registration import register

register(
    id='carpark-v0',
    entry_point='gym_carpark.envs:CarparkEnv',
)
