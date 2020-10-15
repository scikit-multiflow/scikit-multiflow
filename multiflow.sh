#! /bin/bash
set -e # Ensure we fail if an error occurs
set -o pipefail # Ensure we fail if something fails at pipeline level

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
######################## GET SCRIPT LOCATION - END - ################################

PROFILE="${PROFILE:-LOCAL}"
# Possible profiles: "LOCAL"
MODE="$1"

# Check if option passed is valid
OPTIONS_ARRAY=( "--setup" "--help" "--run-tests" "--run-test" )
VALID_OPTION="false"
if [ -z "$1" ]; then
    echo "Please specify an option!"
    IFS=$'\n'; echo "${OPTIONS_ARRAY[*]}"
    exit 1
fi
for k in "${OPTIONS_ARRAY[@]}"; do
    if [ $k = $1 ]; then
        VALID_OPTION="true"
    fi
done
if [ $VALID_OPTION = "false" ]; then
    echo "$1 is not a valid option! Valid options are: "
    IFS=$'\n'; echo "${OPTIONS_ARRAY[*]}"
    exit 1
fi 

#########################################################################
# execute requested action

if [ $MODE = "--setup" ]; then
        pip install -U numpy
        pip install -U Cython
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
fi

if [ $MODE = "--run-tests" ]; then
        python setup.py test
fi

if [ $MODE = "--run-test" ]; then
        pytest --capture=tee-sys $2 --showlocals -v
fi

if [ $MODE = "--help" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        MULTIFLOW_SCRIPT="bash multiflow.sh"
        echo "$MULTIFLOW_SCRIPT --setup                                                              | setup environment"
        echo "$MULTIFLOW_SCRIPT --run-tests                                                          | run all tests"
        echo "$MULTIFLOW_SCRIPT --run-test <TEST-NAME>                                               | run a specific test"
    fi
fi
