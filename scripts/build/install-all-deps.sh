#! /bin/bash
set -e #Ensure we fail if an error occurs
set -o pipefail #Ensure we fail if something fails at pipeline level

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MULTIFLOW_BASEDIR=$SCRIPT_LOCATION/../..
######################## GET SCRIPT LOCATION - END - ################################
cd $MULTIFLOW_BASEDIR

echo "Installing python dependencies from 'requirements.txt' and 'requirements-dev.txt'"
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
