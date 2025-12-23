#!/bin/bash

COUNT=100
NUM_ENVS=10
OUTPUT_DIR="/home/ryo/Downloads/250923"
VISUALIZE=false

if [ "$VISUALIZE" = true ]; then
    VISOPT="--vis"
else
    VISOPT=""
fi

# --use_env_statesと--use_first_env_stateの違いは？
python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
    --sim-backend cpu -o 'state+rgb+ts_force' -b physx_cpu --save-traj $VISOPT --save-video --use-env-states \
    --count $COUNT --render-mode rgb_array --num-envs $NUM_ENVS \
    --output-dir $OUTPUT_DIR

python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
    --sim-backend cpu -o 'state+rgb+ts_force' -b physx_cpu --save-traj $VISOPT --save-video --use-env-states \
    --count $COUNT --render-mode rgb_array --num-envs $NUM_ENVS \
    --output-dir $OUTPUT_DIR

python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
    --sim-backend cpu -o 'state+rgb+ts_force' -b physx_cpu --save-traj $VISOPT --save-video --use-env-states \
    --count $COUNT --render-mode rgb_array --num-envs $NUM_ENVS \
    --output-dir $OUTPUT_DIR

