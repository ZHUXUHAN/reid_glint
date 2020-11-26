cd ~ || exit
git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
cd libjpeg-turbo || exit
mkdir build
cd build || exit
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j48
sudo make install
cd ~ || exit