# mahjong-reinforcement-learning
麻雀の強化学習実験

シミュレータに[mjx](https://github.com/mjx-project/mjx)利用。(version 0.1.0)

# 教師あり学習

`ShantenAgent`(シャンテン数を減らす方向に打つ、基本的なルールベースのエージェント)の行動を教師としてpolicyおよびvalueのDNNを教師あり学習する

```
mkdir -p data
# 教師データ作成
python -m mjxrl.gen_play data/supervise_1_train.pkl.gz 10000
python -m mjxrl.preprocess_sv data/supervise_1_train.pkl.gz data/supervise_1_train_pp.pkl.gz
python -m mjxrl.gen_play data/supervise_1_val.pkl.gz 1000
python -m mjxrl.preprocess_sv data/supervise_1_val.pkl.gz data/supervise_1_val_pp.pkl.gz
# 学習
python -m mjxrl.sv_train_policy data/sv_policy_1 data/supervise_1_train_pp.pkl.gz data/supervise_1_val_pp.pkl.gz
python -m mjxrl.sv_train_value data/sv_policy_1 data/supervise_1_train_pp.pkl.gz data/supervise_1_val_pp.pkl.gz
# 学習したモデルと、ランダムエージェント3人を対局させて勝率を見る
python -m mjxrl.val_policy data/sv_policy_1/ 1000 RandomAgent
```

RandomAgent相手で勝率0.145。誰も上がらない場合がよくあるので、ランダムでも0.25には達しない。全員ランダムだと勝率はほぼ0。
