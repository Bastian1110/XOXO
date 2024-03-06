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
algo.restore("./models/XOXOv1.0/")

env = TicTacToeMultiAgent()
obs = env.reset()
done = False

while not done:
    print("Board :")
    print("Actual Player :", env.current_player)
    print(obs)
    if env.current_player == 1:
        action  = algo.compute_single_action(obs, policy_id="pol1")
        print("Agent Action", action)
    else:
        action = int(input("Enter your action: "))
    obs, reward, terminated, _, info = env.step({env.current_player : action})
    done = terminated[env.current_player]
    print(f"Action: {action}, Reward: {reward}, Info: {info}")