#!/usr/bin/env bash

set -x

videos=$1
direction=$2

for video in ${videos}
do
    log_file=$(basename ${video} | cut -c1-14)
    python vehicle_tracker.py -s ${video} -d ${direction} >> logs/${log_file}_${direction}.log
done
