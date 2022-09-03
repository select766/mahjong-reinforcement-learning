"""
ShantenAgent(学習不要のルールベースエージェント)を用いて、学習データを作成する
"""

import argparse
from collections import defaultdict
import pickle
import gzip
import numpy as np

import mjx
from mjx.agents import ShantenAgent

players = [f"player_{i}" for i in range(4)]


def run_one_game(env: mjx.MjxEnv, agent: ShantenAgent, reward_type: str, done_type: str, feature_type: str):
    obs_dict = env.reset()
    player_obs_tensors = defaultdict(list)  # item: np.ndarray
    player_actions = defaultdict(list)  # item: int
    player_action_masks = defaultdict(list)  # item: np.ndarray
    while not env.done(done_type):
        actions = {}
        for player_id, obs in obs_dict.items():
            action = agent.act(obs)
            actions[player_id] = action
            player_obs_tensors[player_id].append(obs.to_features(feature_type).astype(np.uint8)) # 数値は0/1なので容量削減
            player_action_masks[player_id].append(obs.action_mask().astype(np.uint8)) # 数値は0/1なので容量削減
            player_actions[player_id].append(action.to_idx())
        obs_dict = env.step(actions)
    rewards = env.rewards(reward_type)
    return {player_id: {"obs_hist": player_obs_tensors[player_id], "action_hist": player_actions[player_id], "action_mask_hist": player_action_masks[player_id], "reward": rewards[player_id]} for player_id in players}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    agent = ShantenAgent()
    env = mjx.MjxEnv()
    games = []
    for i in range(args.n):
        games.append(run_one_game(
            env, agent,
            reward_type="round_win",  # 局の勝者に1、それ以外に0
            done_type="round",  # 1局で終了
            # shape=(16,34), dtype=np.int32, 数値は0か1
            feature_type="mjx-small-v0",
        ))
    with gzip.open(args.dst, "wb") as f:
        pickle.dump(games, f)


if __name__ == "__main__":
    main()
