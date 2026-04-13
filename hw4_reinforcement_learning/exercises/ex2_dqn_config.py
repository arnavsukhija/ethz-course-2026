"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # TODO: Tune the following hyperparameters
    # Replace the default values with your own choices.
    "lr": 5e-4,            # TODO: use lower learning rate for stability
    "epsilon": 0.1,       # TODO: 0.03 is too low for exploration, use 0.1
    "target_update": 50,   # TODO: lets increase to 50 to stabilize training
    "hidden_dim": 128,     # TODO: is fine, cartpole has simple state space
    
    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}