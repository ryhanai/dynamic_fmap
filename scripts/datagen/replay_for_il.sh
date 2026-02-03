#!/bin/bash

COUNT=100
NUM_ENVS=10

OUTPUT_DIR="/home/ryo/Downloads/260128"
VISUALIZE=false

if [ "$VISUALIZE" = true ]; then
    VISOPT="--vis"
else
    VISOPT=""
fi


python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
    --use-first-env-state -c pd_ee_delta_pos -o 'state+rgb+ts_force' \
    --save-traj --num-envs $NUM_ENVS -b physx_cpu --sim-backend cpu \
    --save-video $VISOPT --count $COUNT --output-dir $OUTPUT_DIR \
    --render-mode rgb_array

python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
    --use-first-env-state -c pd_ee_delta_pose -o 'state+rgb+ts_force' \
    --save-traj --num-envs $NUM_ENVS -b physx_cpu --sim-backend cpu \
    --save-video $VISOPT --count $COUNT --output-dir $OUTPUT_DIR \
    --render-mode rgb_array

python -m dynamic_fmap.benchmarks.maniskill.replay_trajectory \
    --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
    --use-first-env-state -c pd_ee_delta_pose -o 'state+rgb+ts_force' \
    --save-traj --num-envs $NUM_ENVS -b physx_cpu --sim-backend cpu \
    --save-video $VISOPT --count $COUNT --output-dir $OUTPUT_DIR \
    --render-mode rgb_array
