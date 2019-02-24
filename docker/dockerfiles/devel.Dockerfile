FROM python:3.6.8-jessie

RUN mkdir /app

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN python -m pip install -U numpy
# Clone scikit-multiflow
RUN git clone https://github.com/garawalid/scikit-multiflow.git
# Install scikit-multiflow
RUN cd scikit-multiflow && python -m pip install -U . && cd ..
# Copy examples and dataset
RUN cp scikit-multiflow/docker/examples/src/* .
RUN cp scikit-multiflow/docker/examples/data/* .


# Clean directory
RUN rm -rf scikit-multiflow

CMD /bin/bash
