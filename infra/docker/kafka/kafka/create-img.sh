#!/bin/bash
basename="multiflow/kafka"

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MULTIFLOW_BASEDIR=$SCRIPT_LOCATION/../../../..
######################## GET SCRIPT LOCATION - END - ################################
hash=$(bash $MULTIFLOW_BASEDIR/scripts/build/get-current-version.sh)
imgtag1="$basename:latest"
imgtag2="$basename:$hash"

PUSH="FALSE"
while getopts "p" arg; do
  case $arg in
    p)
      PUSH="TRUE"
      ;;
  esac
done

docker build --pull --no-cache --squash . -t $imgtag1 -t $imgtag2
if [ $PUSH = "TRUE" ]; then
    docker push $imgtag1
    docker push $imgtag2
fi
docker images $imgtag1 --format "{{.Repository}}:{{.Tag}} {{.Size}}"
