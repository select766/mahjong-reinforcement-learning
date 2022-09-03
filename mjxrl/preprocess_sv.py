"""
ゲームプレイデータを学習データに変換する
"""

import argparse
import pickle
import gzip
import numpy as np

players = [f"player_{i}" for i in range(4)]

def preprocess(games):
    all_feats = []
    all_actions = []
    all_action_masks = []
    all_rewards = []
    for record in games:
        for p in players:
            pdata = record[p]
            all_feats.extend(pdata["obs_hist"])
            all_actions.extend(pdata["action_hist"])
            all_action_masks.extend(pdata["action_mask_hist"])
            all_rewards.append(np.full(len(pdata["obs_hist"]), pdata["reward"], dtype=np.float32))
    preprocessed = {
        "obs": np.stack(all_feats), # (n, 16, 34), uint8
        "actions": np.array(all_actions, dtype=np.uint8), # (n, ), uint8
        "action_masks": np.stack(all_action_masks), # (n, 181), uint8
        "discounted_rewards": np.concatenate(all_rewards), # (n, ), float32 割引率を使う可能性があるので実数
    }
    return preprocessed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    with gzip.open(args.src, "rb") as f:
        games = pickle.load(f)
    
    preprocessed = preprocess(games)
    del games

    with gzip.open(args.dst, "wb") as f:
        pickle.dump(preprocessed, f)

if __name__ == "__main__":
    main()
