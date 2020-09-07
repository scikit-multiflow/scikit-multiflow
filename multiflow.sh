#! /bin/bash
set -e # Ensure we fail if an error occurs
set -o pipefail # Ensure we fail if something fails at pipeline level

######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
######################## GET SCRIPT LOCATION - END - ################################

PROFILE="${PROFILE:-LOCAL}"
# Possible profiles: "LOCAL", "DOCKER-DEV"
MODE="$1"

# Check if option passed is valid
OPTIONS_ARRAY=( "--start" "--recreate" "--stop" "--destroy" "--show-args" "--into-dev" "--attach-tmux" "--help" "--update-images" "--purge-images" "--run-tests" "--run-test" )
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

if [ $MODE = "--start" ] || [ $MODE = "--recreate" ] || [ $MODE = "--prepare" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        /bin/bash $SCRIPT_LOCATION/scripts/dev/cli/start-dev-env.sh $SCRIPT_LOCATION $MODE
    else
        echo "Detected $PROFILE profile - $MODE allowed outside Docker only."
    fi
fi
if [ $MODE = "--destroy" ]; then
    if [ $PROFILE != "LOCAL" ]; then
        echo "Detected $PROFILE profile - $MODE allowed outside Docker only."
    fi
fi
if [ $MODE = "--stop" ] || [ $MODE = "--destroy" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        /bin/bash $SCRIPT_LOCATION/scripts/dev/cli/stop-dev-env.sh $SCRIPT_LOCATION $MODE
    else
        echo "Detected $PROFILE profile - $MODE allowed outside Docker only."
    fi
fi
if [ $MODE = "--show-args" ]; then
    if [ $PROFILE != "LOCAL" ]; then
        /bin/bash $SCRIPT_LOCATION/scripts/dev/cli/show-env-variables.sh
    else
        echo "$MODE allowed inside Docker only."
    fi
fi
if [ $MODE = "--into-dev" ]; then
    if [ $PROFILE = "LOCAL" ]; then
            /bin/bash $SCRIPT_LOCATION/scripts/dev/cli/log-into-dev.sh
    else
        echo "Detected $PROFILE profile - $MODE allowed outside Docker only."
    fi
fi
if [ $MODE = "--attach-tmux" ]; then
    if [ $PROFILE != "LOCAL" ]; then
        x=$(tmux list-session | grep 'ql' || true)
        x=$(echo $x |wc -l)
        if [ $x -eq 0 ]; then
            export HOME_DIR=/home/multiflow
            . $HOME_DIR/scripts/common/setup-env-variables.sh
            if [ $PROFILE = "DOCKER-DEV" ]; then
                /bin/bash $SCRIPT_LOCATION/scripts/common/start-scripts/tmux-start-dev.sh
            fi
        fi
        /bin/bash $SCRIPT_LOCATION/scripts/dev/cli/attach-tmux.sh
    else
        echo "$MODE allowed inside Docker only."
    fi
fi
if [ $MODE = "--update-images" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        docker pull "jmrozanec/multiflow-dev:latest"
        docker pull "jmrozanec/kafka:latest"
    else
        echo "$MODE allowed outside Docker only."
    fi
fi

if [ $MODE = "--purge-images" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        docker images -f dangling=true | awk '{print $3}' | xargs docker rmi
    else
        echo "$MODE allowed outside Docker only."
    fi
fi

if [ $MODE = "--rebuild-images" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        allchangescommitted=$(git diff-index --quiet HEAD -- || echo "NO")
        allchangescommitted=${allchangescommitted:-YES}
        if [ $allchangescommitted = "YES" ]; then
            /bin/bash $SCRIPT_LOCATION/scripts/build/build-docker-images-tree.sh -p
        else
            echo "We detected uncommitted changes: please commit before building the Docker images."
        fi
    else
        echo "$MODE allowed outside Docker only."
    fi
fi

if [ $MODE = "--run-tests" ]; then
    if [ $PROFILE != "LOCAL" ]; then
        cd /home/multiflow && python setup.py test
    else
        echo "$MODE allowed inside Docker only."
    fi
fi

if [ $MODE = "--run-test" ]; then
    if [ $PROFILE != "LOCAL" ]; then
        pytest $2 --showlocals -v
    else
        echo "$MODE allowed inside Docker only."
    fi
fi

if [ $MODE = "--help" ]; then
    if [ $PROFILE = "LOCAL" ]; then
        MULTIFLOW_SCRIPT="bash multiflow.sh"
        echo "To START DEV environment"
        echo "    $MULTIFLOW_SCRIPT --start                                                          | create a container, using existing volumes"
        echo "    $MULTIFLOW_SCRIPT --recreate                                                       | recreates containers"
        echo "To STOP DEV environment"
        echo "    $MULTIFLOW_SCRIPT --stop                                                           | stops existing containers with graceful shutdown"
        echo "    $MULTIFLOW_SCRIPT --destroy                                                        | kills existing containers and disks"
        echo "OTHER utilities"
        echo "    $MULTIFLOW_SCRIPT --into-dev                                                       | get into multiflow-dev container"
        echo "    $MULTIFLOW_SCRIPT --update-images                                                  | update local Docker images to latest version"
        echo "    $MULTIFLOW_SCRIPT --rebuild-images                                                 | rebuild our Docker images tree"
        echo "    $MULTIFLOW_SCRIPT --purge-images                                                   | purge Docker dangling images"
        echo "    $MULTIFLOW_SCRIPT --run-tests                                                      | run all tests"
        echo "    $MULTIFLOW_SCRIPT --run-test <TEST-NAME>                                           | run a specific test"
    else
        echo "$MULTIFLOW_SCRIPT --attach-tmux                                                        | attach tmux session"
        echo "$MULTIFLOW_SCRIPT --show-args                                                          | display relevant environment variables"
    fi
fi
