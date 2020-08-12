# Dockerized Kafka 

The following template was taken from https://hub.docker.com/r/flozano/kafka

This repository provides everything you need to run Kafka in Docker. This is a fork
from https://github.com/spotify/docker-kafka. Main changes are:

- removal of legacy kafka proxy
- bumped to new version + fixed archive url
- fixed init bug (missing newline that messes up kafka server config)

Why?
---
The main hurdle of running Kafka in Docker is that it depends on Zookeeper.
Compared to other Kafka docker images, this one runs both Zookeeper and Kafka
in the same container. This means:

* No dependency on an external Zookeeper host, or linking to another container
* Zookeeper and Kafka are configured to work together out of the box

Run
---

```bash
docker run -p 2181:2181 -p 9092:9092 --env ADVERTISED_HOST=$(docker-machine ip $(docker-machine active)) --env ADVERTISED_PORT=9092 yourdockerhub/kafka
```

```bash
export KAFKA=$(docker-machine ip $(docker-machine active)):9092
kafka-console-producer.sh --broker-list $KAFKA --topic test
```

```bash
export KAFKA=$(docker-machine ip $(docker-machine active)):9092
kafka-console-consumer.sh --bootstrap-server $KAFKA --topic test
```
