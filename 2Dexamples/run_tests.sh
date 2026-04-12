#!/bin/bash
 
PLANNERS=("cdf" "bias")
# PLANNERS=("vanilla" "cdf" "bias")
SCENES=("scene_2")
# SCENES=("scene_1" "scene_2" "scene_3" "scene_4" "scene_5")
ITERS=("250" "500" "1000" "2000")
 
for iters in "${ITERS[@]}"; do
    for scene in "${SCENES[@]}"; do
        for planner in "${PLANNERS[@]}"; do
            echo "Running: iters=$iters scene=$scene planner=$planner"
            python ee_goal_rrtstar_2d.py \
                --planner "$planner" \
                --scene "$scene" \
                --max-iters "$iters" \
                --no-talkback
        done
    done
done
 
