#!/usr/bin/env bash

set -x

readonly SCRIPT_DIR="$(dirname $(readlink -f $0))"
readonly APP_HOME="$(realpath ${SCRIPT_DIR}/..)"
readonly LOG_DIR="${APP_HOME}/logs"
readonly SRC_DIR="${APP_HOME}/src/vehicle_tracker"

video_home=$1
direction=$2

for video in ${video_home}/*P000.mp4
do
    log_file="${LOG_DIR}/$(basename ${video} | cut -c1-14)-${direction}.log"
    python ${SRC_DIR}/main.py ${video} ${direction} ${log_file} -i 2 -d
done
