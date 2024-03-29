import ray
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from TicTacToe import TicTacToeMultiAgent
import ray.tune as tune 

tune.register_env("tictactoe_multiagent", lambda config: TicTacToeMultiAgent(config))

def mapping_fn(agent_id, *args, **kwargs):
    return "pol1" if agent_id == 1 else "pol2"

config = (
    PPOConfig()
    .framework("torch")
    .environment("tictactoe_multiagent", disable_env_checking=True)
    .multi_agent(
        policies={
            "pol1" : PolicySpec(config={"gamma": 0.85}),
            "pol2" : PolicySpec(config={"gamma": 0.95}),
        },
        policy_mapping_fn=mapping_fn
    )
    .rollouts(num_rollout_workers=0) 

)

algo = PPO(config)

for i in range(100):
    result = algo.train()
    print(pretty_print(result))

    if i % 10 == 0:
        checkpoint = algo.save("./models/XOXOv1.0")
        print("Checkpoint saved at", checkpoint)