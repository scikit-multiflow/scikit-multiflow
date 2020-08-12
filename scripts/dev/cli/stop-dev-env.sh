#! /bin/bash
set -e # Ensure we fail if an error occurs
set -o pipefail #Ensure we fail if something fails at pipeline level

PROJECT_DIR=$1
MODE=$2

MERGED_CONFIG_FILENAME="$PROJECT_DIR/local-devconfig-files/docker-compose-merged-dev-config.yml"

rm -f tmux-*.log > /dev/null

if [ $MODE = "--stop" ]; then
    docker-compose -f $MERGED_CONFIG_FILENAME stop
fi
if [ $MODE = "--destroy" ]; then
    docker-compose -f $MERGED_CONFIG_FILENAME down --volumes
fi
