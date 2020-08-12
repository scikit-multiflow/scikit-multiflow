#! /bin/bash
set -e
#########################################################################
# This script is an entry-point for execution within docker containers
#########################################################################

echo "INSIDE DOCKER ENTRY-POINT"

export HOME_DIR=/home/multiflow

# SETUP END DISPLAY ENV VARIABLES
# Kudos to: https://stackoverflow.com/questions/496702/can-a-shell-script-set-environment-variables-of-the-calling-shell
#
. $HOME_DIR/scripts/common/setup-env-variables.sh

echo "###########################################################"
echo "#"
echo "#             Welcome to scikit-multiflow development!"
echo "#"
echo "###########################################################"

if [ $PROFILE = "DOCKER-DEV" ]; then
    echo "Installing required dependencies..."
    /bin/bash $HOME_DIR/scripts/build/install-all-deps.sh
    /bin/bash $HOME_DIR/scripts/build/build-all.sh
fi
