#!/bin/bash
SESSIONNAME="multiflow"

cd $HOME_DIR
cat /var/multiflow/templates/.tmux.conf.tpl | sed "s/TMUX_COLOR_01/$TMUX_COLOR_01/g" | sed "s/TMUX_COLOR_02/$TMUX_COLOR_02/g" > $HOME_DIR/.tmux.conf

tmux_session_exists=$(tmux has-session -t $SESSIONNAME |& tee -a $NOHUP_FILE)
tmux_session_exists=${tmux_session_exists:-0}

# 0 | 1
# 4 | 5
# 2 | 3
# 6 | 7/8

#https://unix.stackexchange.com/questions/375567/tmux-after-split-window-how-do-i-know-the-new-pane-id
get_pane_id() {
    tmux list-panes -F '#{pane_id}' -t "$current_window_id" | sort -n --key=1.2 | tail -1
}

if [ "$tmux_session_exists" != "0" ]; then
    tmux new-session -s $SESSIONNAME -n dev -d -c $HOME_DIR/src/cli                                                                 #0
    tmux source-file $HOME_DIR/.tmux.conf

    P0=$(get_pane_id)
    tmux send-keys -t $SESSIONNAME "ls" C-m

tmux_session_exists=$(tmux has-session -t $SESSIONNAME |& tee -a $NOHUP_FILE)
tmux_session_exists=${tmux_session_exists:-0}

if [ "$tmux_session_exists" == "0" ]; then
    echo "TMUX ready!"
else
    echo "'$SESSIONNAME' tmux session not found!"
fi
