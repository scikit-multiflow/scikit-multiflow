#!/bin/bash

PUSH="FALSE"
while getopts "p" arg; do
  case $arg in
    p)
      PUSH="TRUE"
      ;;
  esac
done

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MULTIFLOW_BASEDIR=$SCRIPT_LOCATION/../..
echo "Multiflow basedir is '$MULTIFLOW_BASEDIR'"
######################## GET SCRIPT LOCATION - END - ################################

DOCKER_BASEDIR=$MULTIFLOW_BASEDIR/infra/docker
PUSH_PARAM=""
if [ $PUSH = "TRUE" ]; then
    PUSH_PARAM="-p"
fi

echo "Building multiflow-dev ..." && cd $DOCKER_BASEDIR/multiflow-dev && bash create-img.sh $PUSH_PARAM
echo "Building multiflow-kafka ..." && cd $DOCKER_BASEDIR/multiflow-kafka && bash create-img.sh $PUSH_PARAM
