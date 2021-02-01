# vrp_config = {
#     "problem": "A-n32-k5",
#     "episode_len": 3200
# }

vrp_config = {
    "problem": "E-n101-k14",
    "episode_len": 100
}

vrp_dqn_config = {
    "problem": "E-n101-k14",
    "episode_len": 100,
    "action_space_size": 4*4 + 1
}

knapsack_config = {
    "problem": "p08",
    'episode_len': 24
}


knapsack_dqn_config = {
    "problem": "p08",
    'episode_len': 24,
    'training_timestep': 100
}