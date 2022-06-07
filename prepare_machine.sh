#!/bin/bash

CONDA_ENV_NAME="omni-anynet"
PYTORCH_VERSION="1.12" # does not work with PyTorch 2.x

abortOnFailure() {
    status=$?
    if [ "$status" -ne 0 ];
    then
        echo "ERROR: $status" 1>&2;
        exit $status
    fi
}

# Check for submodules

if ! [ -f "Omni_lut/main_Pytorch.py" ] || ! [ -f "map_processing/map_converter.py" ];
then
    git submodule update --init --recurse
fi


# Check for pretrained SceneFlow model

if ! [ -f "checkpoint/sceneflow/sceneflow.tar" ];
then
    echo "Please download the sceneflow model from the AnyNet webpage" 2>&1
    echo "https://github.com/mileyan/AnyNet" 2>&1
    echo "Target: checkpoint/sceneflow/sceneflow.tar" 2>&1
    exit 1
fi


# Check for required debian packages

[ -z "$CUDA_VERSION" ] && CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \(.*\),.*/\1/g')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -f 1 -d".")
CUDA_MINOR=$(echo $CUDA_VERSION | cut -f 2 -d".")

# check for required packages
if [ "$CUDA_MAJOR" -eq "10" ];
then
    gnuc_version=8
elif [ "$CUDA_MAJOR" -eq "11" ];
then
    if [ "$CUDA_MINOR" -eq "0" ];
    then
        gnuc_version=9
    elif [ "$CUDA_MINOR" -le "4" ];
    then
        gnuc_version=10
    else
        gnuc_version=11
    fi
else
    echo "Please fix the script to set the gnu compiler version." 2>&1;
    exit 1
fi

c_bin="gcc-$gnuc_version"
cxx_bin="g++-$gnuc_version"
export CC="/usr/bin/$c_bin"
export CXX="/usr/bin/$cxx_bin"

REQ_ERROR=false
# Hint: Pip's openexr with system's libs openexr and libopenexr-dev is much faster than condas openexr-python!
req_packages=( 'openexr' 'libopenexr-dev' 'python3-dev' 'zlib1g-dev' 'libjpeg-turbo8' "$c_bin" "$cxx_bin" )
for req in ${req_packages[@]};
do
    if ! dpkg -l | grep " ${req}" 2>&1 > /dev/null;
    then
        echo "Error: The package $req is missing. On Ubuntu you can install it with:"
        echo "sudo apt install ${req}"
        REQ_ERROR=true
    fi
done

if [ $REQ_ERROR == true ];
then
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh || exit $?
if [ $(conda env list | grep "$CONDA_ENV_NAME " | wc -l) -eq 0 ];
then
    conda create -n $CONDA_ENV_NAME python=3.10.4 -y
fi    
conda activate $CONDA_ENV_NAME || exit $?


# Install required Python packages into conda env
    
if [ "$CUDA_MAJOR" -eq "10" ];
then
    conda install "pytorch=$PYTORCH_VERSION" torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
elif [ "$CUDA_MAJOR" -eq "11" ];
then
    if [ "$CUDA_MINOR" -ge "6" ];
    then
        conda install "pytorch=$PYTORCH_VERSION" torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
    else
        conda install "pytorch=$PYTORCH_VERSION" torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    fi  
else
    echo "Please fix the PyTorch cuda verion within this script." 2>&1
    exit 1
fi

if ! python -c "import torch; exit(1) if not torch.cuda.is_available() else exit(0);";
then
    echo "ERROR: CUDA is not available for PyTorch in your conda environment $CONDA_ENV_NAME." 2>&1
    exit 1;
fi

abortOnFailure
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing natsort matplotlib tifffile tqdm && \
conda install -y -c conda-forge tensorboardX typing_extensions && \
python -m pip install Pillow OpenEXR==1.3.8 Imath==0.0.2 # Do not use OpenEXR 1.3.9 (buggy)!
# ==7.2.0
abortOnFailure

# Compile SPN

cd models/spn_t1
bash make.sh
abortOnFailure
cd gate_lib
ln -s gaterecurrent2dnoind_cuda.*.so gaterecurrent2dnoind_cuda.so
cd ..
cd ../..
    

# Generate initial LUTs and masks

python Omni_lut/main_Pytorch.py
abortOnFailure

echo ""
echo "Preparations finished!"
