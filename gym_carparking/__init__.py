from gym.envs.registration import register

register(
    id='carparking-v0',
    entry_point='gym_carparking.envs:CarparkingEnv',
)
