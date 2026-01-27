# README.md
# dynamic_fmap

A Python module for Imitation Learning with 3D task space force prediction.


## Installation
not yet implemented（ManiSkill3とManiSkill3のdiffusion policyをインストールする）
```bash
pip install .
```

## Usage

### Dataset generation
```bash
$ conda activate diffusion-policy-ms
$ scripts/data_gen/replay_for_il.sh
```
生成するtrajectory数，出力先ディレクトリはこのスクリプト内で定義されている．

### Training
```bash
$ conda activate diffusion-policy-ms
$ cd <dynamic_fmap_root>
$ scripts/train/train_teacher_policy_with_force.sh --seed=0 --gpu=0 --task=PegInsertionSide-v1 --force
```

### Rollout


### Drawing trajectories using graps


### (Optional) Analyze the data range of point forces (coordinates and force magnitude)
```bash
$ conda activate diffusion-policy-ms
$ cd <dynamic_fmap_dir>
$ python -m dynamic_fmap.benchmarks.utils.analyze_trajectory --demo_root <demo_root>
```


## Official command from ManiSkill

### Dataset generation
```bash

```
### Training
```bash

```

## Code structure
```
LICENSE
README.md
pyproject.toml
dynamic_fmap
 |-benchmarks/maniskill（benchmarkごとに定義される追加機能）
  |-envs.py: sensor dataにpoint forcesを追加したEnvクラスを定義
  |-replay_trajectory.py: trajectoryをreplayしてpoint forcesを追加
  |-patch_to_maniskill.py: originalのcodeには手を加えずにmaniskill本体の関数を上書き（observation modeの追加など）
  |-dataset.py
 |-dataset
  |-ForcePredictionDataset.py: not used，monolithicなtrainig programのtraining data管理機能を分割してここに入れる予定
 |-model
 |-agents (policyから名前変更)
  |-train_with_force.py: これをdataset，model, policyに分割し，train_with_force.py自体はtop-levelでも良いかも．
 |-utils
  |-analyze_trajectory.py: trajectoryの分析（data distributionなど）
 |-main.py: not yet used
scripts: scripts for training etc.
tests: test cases
samples: sample programs for the development
```

<!--
    Monkey Patches
    envs/scene.py
    envs/tasks/tabletop/peg_insertion_side.py
    固定長のpoint forcesの場合は不要
-->