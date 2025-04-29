#!/bin/bash
set -ex

PYTHON_VERSION=$1
CUDA_VERSION=$2

if [ ! -z $CUSTOM_PYTHON_VERSION ]; then
    PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
fi

if [ ! -z $CUSTOM_CUDA_VERSION ]; then
    CUDA_VERSION=$CUSTOM_CUDA_VERSION
fi

if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.10"
fi

if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="12.4"
fi

PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

ROOT_PATH=$(pwd)
OUTPUT_PATH=$ROOT_PATH/output
# 获取当前分支名，并将特殊字符转换为下划线
BUILD_TIME=$(date +%Y%m%d%H%M)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
echo "BRANCH_NAME: $BRANCH_NAME"

# 如果分支是以 release_ 或 release/ 开头，则将 release_ 或 release/ 替换为空
if [[ $BRANCH_NAME =~ ^release[\/_] ]]; then
   echo "release branch"
   BRANCH_NAME=${BRANCH_NAME#release}
   BRANCH_NAME=${BRANCH_NAME#/}
   BRANCH_NAME=${BRANCH_NAME#_}
   # 如果分支里还有 / ，则将 / 替换为 .
   BRANCH_NAME=${BRANCH_NAME//\//.}
   if [[ ! -z $BRANCH_NAME ]]; then
       BRANCH_NAME=.${BRANCH_NAME}
   fi
   VERSION_SUFFIX=+byted${BRANCH_NAME}.${BUILD_TIME}
else
   echo "not release branch"
   VERSION_SUFFIX=+byted.${BUILD_TIME}
fi

echo "VERSION_SUFFIX: $VERSION_SUFFIX"

ENABLE_SM90A=$(( ${CUDA_VERSION%.*} >= 12 ? ON : OFF ))

if [ ${CUDA_VERSION} = "12.8" ]; then
   DOCKER_IMAGE="iaas-gpu-cn-beijing.cr.volces.com/pytorch/manylinux2_28-builder:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION//.}"
else
   DOCKER_IMAGE="iaas-gpu-cn-beijing.cr.volces.com/pytorch/manylinux-builder:cuda${CUDA_VERSION}"
   TORCH_INSTALL="pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.}"
fi

# 由于 SCM 的特殊配置，挂载了物理机上的路径到容器内的 /sgl_kernel 目录，所以这里需要拷贝源码到 /sgl_kernel 目录下
BASE_PATH=/sgl_kernel
SRC_PATH=$BASE_PATH/sgl-kernel && mkdir -p $SRC_PATH

rm -rf $SRC_PATH/* && cp -r $(pwd)/sgl-kernel/* $SRC_PATH/
cd $SRC_PATH
VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)
echo "Building sglang-python version $VERSION$VERSION_SUFFIX"
sed -i "s/^version = .*$/version = \"$VERSION$VERSION_SUFFIX\"/" pyproject.toml

docker run --rm \
   -v $(pwd):/sgl-kernel \
   --network=host \
   -e http_proxy=$http_proxy \
   -e https_proxy=$https_proxy \
   ${DOCKER_IMAGE} \
   bash -c "
   # Install CMake (version >= 3.26) - Robust Installation
   export CMAKE_VERSION_MAJOR=3.31
   export CMAKE_VERSION_MINOR=1
   echo \"Downloading CMake from: https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz\"
   wget https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
   tar -xzf cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
   mv cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64 /opt/cmake
   export PATH=/opt/cmake/bin:\$PATH

   # Debugging CMake
   echo \"PATH: \$PATH\"
   which cmake
   cmake --version

   ${PYTHON_ROOT_PATH}/bin/${TORCH_INSTALL} && \
   ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \
   export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX' && \
   export CUDA_VERSION=${CUDA_VERSION} && \
   mkdir -p /usr/lib/x86_64-linux-gnu/ && \
   ln -s /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
   cd /sgl-kernel && \
   ls -la ${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages/wheel/ && \
   PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation -v && \
   ./rename_wheels.sh
   "

# 产物放到 output 目录下
mkdir -p $OUTPUT_PATH
cp -r $SRC_PATH/dist/* $OUTPUT_PATH/
