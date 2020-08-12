#!/bin/bash
export NOHUP_FILE=/tmp/nohup.out

export PROFILE=${PROFILE:-"LH-DEV"}
export PROFILE=$(echo $PROFILE | sed 's/"//g')

. $HOME_DIR/scripts/common/start-scripts/load-tmux-colors.sh

echo "###############################################"
echo "############ ENV VARIABLES - START ############"
printenv | sort
echo "############ ENV VARIABLES - END ##############"
echo "###############################################"

BASHRC=/tmp/.bashrc

echo "NOHUP_FILE=$NOHUP_FILE" >> $BASHRC
echo "PROFILE=$PROFILE" >> $BASHRC
echo "ENVSETTINGS_ENV=$ENVSETTINGS_ENV" >> $BASHRC

echo "TMUX_COLOR_01=$TMUX_COLOR_01" >> $BASHRC
echo "TMUX_COLOR_02=$TMUX_COLOR_02" >> $BASHRC

echo "TZ=$TZ" >> $BASHRC
echo "USER_HOME=$USER_HOME" >> $BASHRC

echo "BASEDIR=$BASEDIR" >> $BASHRC
echo "HOME=$HOME" >> $BASHRC
echo "HOME_DIR=$HOME_DIR" >> $BASHRC
echo "HOSTNAME=$HOSTNAME" >> $BASHRC
echo "PATH=$PATH" >> $BASHRC
echo "PWD=$PWD" >> $BASHRC

cp $BASHRC $HOME_DIR/
