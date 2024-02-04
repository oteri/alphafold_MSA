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
RUN apt-get update && apt-get install -y curl git    \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create the working directory
WORKDIR /app/alphafold_MSA

RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git
ENV REFORMAT_PATH /app/alphafold_MSA/hh-suite/scripts/reformat.pl

# Download and install micromamba
RUN cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN cd /tmp && mv bin/micromamba /usr/local/bin/  && rm -rf bin/

# COPY invalidates always the cache, so t must be at the end.
COPY . /app/alphafold_MSA
RUN micromamba create --prefix /app/env/ -y -f af2_msa.yaml

RUN cp /app/env/bin/ptxas /usr/local/bin/
