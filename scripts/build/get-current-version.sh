######################## GET SCRIPT LOCATION - START - ##############################
SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MULTIFLOW_BASEDIR=$SCRIPT_LOCATION/../..
######################## GET SCRIPT LOCATION - END - ################################
GIT_VERSION=$(git describe --always --abbrev=0)
GIT_VERSION_HASH_LONG=$(git rev-list -n 1 $GIT_VERSION)
GIT_VERSION_HASH_SHORT=$(git rev-parse --short $GIT_VERSION_HASH_LONG)
GIT_SHORT=$(git rev-parse --short HEAD)
VERSION="$GIT_VERSION-$GIT_VERSION_HASH_SHORT-$GIT_SHORT"
echo $VERSION
