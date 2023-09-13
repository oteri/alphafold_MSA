#/bin/bash
set -e 
set -x

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba /usr/local/bin/

mkdir /app
git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git  #needed only for reformat.pl

CUDA_VERSION=$(ls  /usr/local/cuda/lib64/libcudart.so*|tail -n1| xargs -I{} basename {}| sed s/libcudart.so.//g)
CUDA_VERSION=$(echo $CUDA_VERSION|cut -d"." -f1)

micromamba create --prefix /app/env/  -y -c conda-forge \
         openmm=7.7.0 \
         cudatoolkit \
         pdbfixer \
         pip \
         python=3.8 \
      

micromamba run -p /app/env/ pip3 install matplotlib #To use empty_placeholder_template_features ( TO DO: Must be removed) 
micromamba run -p /app/env/ pip3  install -r  requirements.txt --no-cache-dir
micromamba run -p /app/env/ pip3 install --upgrade --no-cache-dir \
      jax==0.3.25 \
      jaxlib==0.3.25+cuda11.cudnn805 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


micromamba install  -p /app/env/  -y   -c nvidia cuda-nvcc=11
cp /app/env/bin/ptxas /usr/local/bin/


# # Check if PARAM_DIR is unset or empty
# if [ -z "$PARAM_DIR" ]; then
#     PARAM_DIR="/data/params"
# fi

# # Check if the directory exists, and create it if it doesn't
# if [ ! -d "$PARAM_DIR" ]; then
#     mkdir -p "$PARAM_DIR"
#     echo "Directory $PARAM_DIR created."
#     cd $PARAM_DIR
#     wget -qO- https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar| tar xf - --no-same-owner
# fi