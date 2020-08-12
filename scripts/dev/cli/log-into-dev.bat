@echo off
for /f %%i in ('docker ps -aqf ancestor^=jmrozanec/multiflow-dev') do set VAR=%%i
docker exec -i -t %VAR% /bin/bash //home/multiflow/multiflow.sh --attach-tmux
