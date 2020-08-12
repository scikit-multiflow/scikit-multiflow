FROM python:3.6.8-jessie
MAINTAINER Joze Rozanec <jmrozanec@gmail.com>

ARG user=multiflow
ARG group=multiflow
ARG uid=1000
ARG gid=1000
ARG USER_HOME="/home/${user}"

ENV USER_HOME ${USER_HOME}

RUN groupadd -g ${gid} ${group} && useradd -d "${USER_HOME}" -u "${uid}" -g "${gid}" -m -s /bin/bash "${user}"
RUN echo "multiflow:multiflow" | chpasswd

# setup SSH server
RUN apt-get update && apt-get install --no-install-recommends -y openssh-server gawk && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN sed -i /etc/ssh/sshd_config \
        -e 's/#PermitRootLogin.*/PermitRootLogin no/' \
        -e 's/#RSAAuthentication.*/RSAAuthentication no/'  \
        -e 's/#PasswordAuthentication.*/PasswordAuthentication yes/' \
        -e 's/#SyslogFacility.*/SyslogFacility AUTH/' \
        -e 's/#LogLevel.*/LogLevel INFO/' && \
    mkdir /var/run/sshd

# SSH login fix. Otherwise user is kicked off after login
# https://docs.docker.com/engine/examples/running_ssh_service/
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

WORKDIR "${USER_HOME}"
#######################################################################################################################################
ENV BASEDIR=/var/multiflow

COPY setup-sshd /usr/local/bin/setup-sshd
RUN chmod 777 /usr/local/bin/setup-sshd && chmod 777 /etc/environment

RUN mkdir -p ${BASEDIR}/templates
ADD tmux.conf.tpl ${BASEDIR}/templates/.tmux.conf.tpl


ENTRYPOINT ["setup-sshd"]
