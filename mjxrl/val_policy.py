"""
学習したpolicy networkをルールベースのエージェントと対局させて勝率を測定する
"""

"""
教師あり学習
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mjx.agents
from .env import GymEnv

agent_classes = {"RandomAgent": mjx.agents.RandomAgent, "ShantenAgent": mjx.agents.ShantenAgent}


class DNNAgent:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def act(self, observation, action_mask):
        with torch.no_grad():
            observation = torch.from_numpy(observation).flatten().float()
            mask = torch.from_numpy(action_mask)
            logits = self.model(observation)
            logits -= (1 - mask) * 1e9
            action = torch.argmax(logits)
            assert action_mask[action.item()] == 1, action_mask[action.item()]
            return int(action.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("n", type=int, help="number of games")
    parser.add_argument("opponent_agent", help="opponent agent type (RandomAgent, ShantenAgent)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    model = nn.Sequential(
        nn.Linear(16 * 34, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 181)
    )
    model.load_state_dict(torch.load(model_dir / "policy.pt"))

    opponent_agent = agent_classes[args.opponent_agent]()
    env = GymEnv(
        opponent_agents=[opponent_agent, opponent_agent, opponent_agent],
        reward_type="round_win",
        done_type="round",
        feature_type="mjx-small-v0",
    )
    agent = DNNAgent(model)

    total_R = 0
    for i in range(args.n):
        obs, info = env.reset()
        done = False
        R = 0
        while not done:
            a = agent.act(obs, info["action_mask"])
            obs, r, done, info = env.step(a)
            R += r
        total_R += R
    # 勝った回数の平均。他人が上がった場合、誰も上がらなかった場合どちらもR=0のため、全員がランダムに行動すれば0.25ではなくほぼ0になる。
    print(f"average reward: {total_R / args.n}")


if __name__ == "__main__":
    main()
