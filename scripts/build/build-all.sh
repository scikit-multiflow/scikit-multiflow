#! /bin/bash
set -e # Ensure we fail if an error occurs
set -o pipefail #Ensure we fail if something fails at pipeline level

VARX=$1
VARX="${VARX:-0}"

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MULTIFLOW_BASEDIR=$SCRIPT_LOCATION/../..
######################## GET SCRIPT LOCATION - END - ################################

# TODO: complete
#bash $SCRIPT_LOCATION/build-multiflow.sh
