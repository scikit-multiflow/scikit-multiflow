#!/bin/bash
docker exec -i -t $(docker ps -aqf ancestor=multiflow/multiflow-dev) /bin/bash
