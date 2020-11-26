export USE_BYTESCHEDULER=1
export BYTESCHEDULER_WITH_MXNET=1
export BYTESCHEDULER_WITHOUT_PYTORCH=1
export MXNET_ROOT=/root/incubator-mxnet

# Install gcc 4.9
mkdir -p ~/gcc/ && cd ~/gcc || exit

cd ~/gcc &&\
sudo dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i libmpfr4_3.1.4-1_amd64.deb &&\
sudo dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
sudo dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

# Pin GCC to 4.9 (priority 200) to compile correctly against MXNet
sudo update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
sudo update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

# May need to uninstall default MXNet and install mxnet-cu90==1.5.0

# Clone MXNet as ByteScheduler compilation requires header files
cd ~ || exit
git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet && git reset --hard 75a9e187d00a8b7ebc71412a02ed0e3ae489d91f

# Install ByteScheduler
pip install bayesian-optimization
git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install
