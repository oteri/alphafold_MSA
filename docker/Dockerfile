# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y wget git    \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite

# Download and install micromamba
RUN cd /tmp && wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN cd /tmp && mv bin/micromamba /usr/local/bin/  && rm -rf bin/

# Create a conda environment
RUN micromamba create --prefix /app/env/ -y -c conda-forge \
    matplotlib \
    openmm=7.7.0 \
    cudatoolkit \
    pdbfixer \
    pip \
    python=3.8

# Create the working directory
COPY . /app/alphafold_MSA

WORKDIR /app/alphafold_MSA

RUN mv  /tmp/hh-suite  .
ENV REFORMAT_PATH /app/alphafold_MSA/hh-suite/scripts/reformat.pl

RUN micromamba run -p /app/env/ pip install -r requirements.txt --no-cache-dir

# Install specific versions of JAX and JAXlib
RUN micromamba run -p /app/env/ pip install --upgrade --no-cache-dir \
    jax==0.3.25 \
    jaxlib==0.3.25+cuda11.cudnn805 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install CUDA nvcc compiler
RUN micromamba install -p /app/env/ -y -c nvidia cuda-nvcc=11

# Copy ptxas to system path
RUN cp /app/env/bin/ptxas /usr/local/bin/
