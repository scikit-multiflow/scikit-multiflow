#! /bin/bash
set -e # Ensure we fail if an error occurs
set -o pipefail #Ensure we fail if something fails at pipeline level
ECHO_PREFIX="### [DEVELOPMENT]"

PROJECT_DIR=$1
MODE=$2
echo "$ECHO_PREFIX Start mode: '$MODE'"

os_name=$(uname -s | cut -c-4 | awk '{print tolower($0)}')
windows_names="ming msys wind uwin cygw ms-d inte"
replaced_os_name=$(echo $windows_names | sed  "s#$os_name#xXxXxXxX#" )
current_os="linux"
if [ ${#windows_names} -eq ${#replaced_os_name} ]; then
    current_os="linux"
else
    current_os="windows"
fi

echo "$ECHO_PREFIX Using project location '$PROJECT_DIR'"

echo "Check if required Docker images are locally available."
if [[ "$(docker images -q multiflow/multiflow-dev | wc -l | xargs)" == "0" ]]; then
    cd $PROJECT_DIR/infra/docker/multiflow-dev/ && bash create-img.sh && cd $PROJECT_DIR
fi
if [[ "$(docker images -q multiflow/kafka | wc -l | xargs)" == "0" ]]; then
    cd $PROJECT_DIR/infra/docker/kafka/kafka/ && bash create-img.sh && cd $PROJECT_DIR
fi



CONFIG_DIR=$PROJECT_DIR/infra/compose/compose-files
echo "$ECHO_PREFIX Using config location  '$CONFIG_DIR'"

echo "$ECHO_PREFIX Check and install merge-yaml-cli if not installed"
MERGE_YAML_INSTALLED=$(npm list -g | grep merge-yaml-cli || true | wc -l | awk '{$1=$1;print}')
if [ "$MERGE_YAML_INSTALLED" = '0' ]; then
    npm -g install merge-yaml-cli@1.1.2
else
    echo "$ECHO_PREFIX Package already installed :)"
fi
echo "$ECHO_PREFIX Check and install python-rsa if not installed"
PYTHON_RSA_INSTALLED=$(pip freeze | grep rsa || true | wc -l | awk '{$1=$1;print}')
if [ "$PYTHON_RSA_INSTALLED" = '0' ]; then
    pip install --user rsa==4.0
else
    echo "$ECHO_PREFIX Package already installed :)"
fi
echo "$ECHO_PREFIX Check and install click if not installed"
PYTHON_CLICK_INSTALLED=$(pip freeze | grep click || true | wc -l | awk '{$1=$1;print}')
if [ "$PYTHON_CLICK_INSTALLED" = '0' ]; then
    pip install --user Click==7.0
else
    echo "$ECHO_PREFIX Package already installed :)"
fi

echo "$ECHO_PREFIX Generating unified docker-compose definition ..."
TMP_DIR=$PROJECT_DIR/local-devconfig-files
mkdir -p $TMP_DIR
echo "$ECHO_PREFIX using temporary directory '$TMP_DIR'"
TMP_FILE="$TMP_DIR/docker-compose-merged-dev-config-tmp.yml"
MERGED_CONFIG_FILENAME="$TMP_DIR/docker-compose-merged-dev-config.yml"
merge-yaml -i $CONFIG_DIR/docker-compose.yml $CONFIG_DIR/docker-compose.dev.yml $CONFIG_DIR/docker-compose.nginx-dev.yml -o $TMP_FILE
if [ $current_os == "windows" ]; then
    echo "$ECHO_PREFIX setting up configs for WINDOWS"
    PROJECT_DIR_WINDOWS=$(echo $PROJECT_DIR | sed 's#/##' | awk '{print substr($0,1,1) ":" substr($0,2)}')
    cat $TMP_FILE | awk -v PROJECT_DIR_WINDOWS=$PROJECT_DIR_WINDOWS '{if (($0 ~ /source/) || ($0 ~ /file/))  {c=PROJECT_DIR_WINDOWS; gsub("MULTIFLOW_HOME", c, $0); gsub("/", "\\", $0); print $0} else {print $0}}'  > $MERGED_CONFIG_FILENAME
else
    echo "$ECHO_PREFIX setting up configs for LINUX/MAC"
    cat $TMP_FILE | sed "s#MULTIFLOW_HOME#$PROJECT_DIR#" > $MERGED_CONFIG_FILENAME
fi
rm $TMP_FILE

echo "$ECHO_PREFIX Created docker compose configuration file in '$MERGED_CONFIG_FILENAME'"

if [ $MODE = "--start" ]; then
    echo "$ECHO_PREFIX Starting DEV environment ..."
    COMPOSE_HTTP_TIMEOUT=240 docker-compose -f $MERGED_CONFIG_FILENAME up &
fi

if [ $MODE = "--recreate" ]; then
    echo "$ECHO_PREFIX Starting DEV environment ..."
    docker-compose -f $MERGED_CONFIG_FILENAME up --force-recreate &
fi
