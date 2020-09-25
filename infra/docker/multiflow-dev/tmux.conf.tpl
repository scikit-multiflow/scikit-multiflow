set-option -g mouse on
# make scrolling with wheels work
bind -n WheelUpPane if-shell -F -t = "#{mouse_any_flag}" "send-keys -M" "if -Ft= '#{pane_in_mode}' 'send-keys -M' 'select-pane -t=; copy-mode -e; send-keys -M'"
bind -n WheelDownPane select-pane -t= \; send-keys -M

set -g pane-border-style fg=TMUX_COLOR_01
set -g pane-active-border-style "bg=default fg=TMUX_COLOR_02"
