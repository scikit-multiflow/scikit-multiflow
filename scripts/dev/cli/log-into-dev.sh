#!/bin/bash
docker exec -i -t $(docker ps -aqf ancestor=jmrozanec/multiflow-dev) /bin/bash
