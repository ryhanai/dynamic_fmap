#!/bin/bash

DEMOS=100
DEMO_ROOT="~/Downloads/250923"
MAX_EVAL_EVNS=10
DIFFUSION_POLICY_ROOT="$HOME/Program/ManiSkill/examples/baselines/diffusion_policy"


# --- default values ---
SEED=""
GPU=""
TASK=""
WITH_FORCE=false


usage() {
  echo "Usage: $0 --seed=<int> --gpu=<int> --task=<string>"
  echo "  --seed : positive integer (>=1)"
  echo "  --gpu  : integer from 0 to (num_gpus-1)"
  echo "  --task : arbitrary string (required)"
  echo "  --force: use 3D force PI in observations (optional flag)"
  exit 1
}

# --- 正の整数チェック ---
is_positive_int() {
  case "$1" in
    ''|*[!0-9]*)
      return 1 ;;
    *)
      [ "$1" -gt 0 ] 2>/dev/null
      return $? ;;
  esac
}

# --- 引数解析 ---
while [ $# -gt 0 ]; do
  case "$1" in
    --seed=*)
      SEED="${1#*=}"
      ;;
    --gpu=*)
      GPU="${1#*=}"
      ;;
    --task=*)
      TASK="${1#*=}"
      ;;
    --force)
      WITH_FORCE=true
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
  shift
done

# --- 必須チェック ---
[ -z "$SEED" ] && { echo "Error: --seed is required"; usage; }
[ -z "$GPU" ]  && { echo "Error: --gpu is required"; usage; }
[ -z "$TASK" ] && { echo "Error: --task is required"; usage; }

# --- seedチェック ---
if ! is_positive_int "$SEED"; then
  echo "Error: seed must be a positive integer: '$SEED'"
  exit 1
fi

# --- GPU 台数取得 ---
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  echo "Error: nvidia-smi not found. Cannot determine number of GPUs."
  exit 1
fi

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "Error: No GPUs found."
  exit 1
fi

# --- GPUが整数か確認 ---
case "$GPU" in
  ''|*[!0-9]*)
    echo "Error: gpu index must be an integer: '$GPU'"
    exit 1 ;;
esac

# --- GPU 範囲チェック ---
MAX_INDEX=$(expr "$NUM_GPUS" - 1)

if [ "$GPU" -lt 0 ] || [ "$GPU" -gt "$MAX_INDEX" ]; then
  echo "Error: gpu index must be between 0 and $MAX_INDEX (host has $NUM_GPUS GPUs)"
  exit 1
fi


if [ "$WITH_FORCE" = true ]; then
  TRAINER="$HOME/Program/moonshot/dynamic_fmap/dynamic_fmap/policy/train_with_force.py"
else
  TRAINER="$HOME/Program/ManiSkill/examples/baselines/diffusion_policy/train.py"
fi


# --- Configuration ---
echo "TRAINER     : $TRAINER"
echo "NUM DEMOS   : $DEMOS"
echo "TASK        : $TASK"
echo "HOST GPUS   : $NUM_GPUS"
echo "SEED        : $SEED"
echo "GPU         : $GPU"


export CUDA_VISIBLE_DEVICES="$GPU"
seed="$SEED"
demos="$DEMOS"

# --- 実行コマンドを変数に入れる ---
case "$TASK" in
  StackCube-v1)
    CMD="python $TRAINER --env-id $TASK \
        --demo-path $DEMO_ROOT/$TASK/motionplanning/trajectory.state+rgb.pd_ee_delta_pos.physx_cpu.h5 \
        --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 200 \
        --total_iters 30000 \
        --obs-mode "state+rgb" \
        --exp-name diffusion_policy-${TASK}-state+force-${demos}_motionplanning_demos-${seed} \
        --demo_type=motionplanning --track"
    ;;
  PegInsertionSide-v1)
    CMD="python $TRAINER --env-id $TASK \
        --demo-path $DEMO_ROOT/$TASK/motionplanning/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5 \
        --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 300 \
        --total_iters 200000 \
        --obs-mode "state+rgb" \
        --exp-name diffusion_policy-${TASK}-v1-state+force-${demos}_motionplanning_demos-${seed} \
        --demo_type=motionplanning --track"
    ;;
  PushT-v1)
    CMD="python $TRAINER --env-id $TASK \
        --demo-path $DEMO_ROOT/$TASK/rl/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5 \
        --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 150 --num_eval_envs $MAX_EVAL_ENVS \
        --total_iters 100000 --act_horizon 1 \
        --obs-mode "state+rgb" \
        --exp-name diffusion_policy-${TASK}-state+force-${demos}_rl_demos-${seed} --no_capture_video \
        --demo_type=rl --track"
    ;;
  *)
    echo "Error: Unknown task '$TASK'"
    echo "Supported: StacuCube-v1 / PegInsertionSide-v1 / PushT-v1"
    exit 1
    ;;
esac

# --- 実行 ---
# ManiSkill付属のdiffusion_policyはimportするパッケージとして作られていないので，
# diffusion_policyのディレクトリに移動してから実行する
pushd $DIFFUSION_POLICY_ROOT
eval $CMD
popd


# for seed in {1..1}; do
#     echo "seed = $seed"
#     python $TRAINER \
#         --dataset_path ~/Dataset/forcemap\
#         --task_name tabletop250902\
#         --model fmdev.force_estimation_v4.ForceEstimationResNetTabletop\
#         --epoch 200\
#         --batch_size 16\
#         --seed $seed\
#         --lr 1e-3\
#         --method 'geometry-aware'\
#         --sigma_f 0.03\
#         --sigma_g 0.01
# done
