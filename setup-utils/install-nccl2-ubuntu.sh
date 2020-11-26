#!/bin/bash
#wget https://developer.download.nvidia.cn/compute/machine-learning/nccl/secure/v2.6/prod/nccl-repo-ubuntu1604-2.6.4-ga-cuda10.0_1-1_amd64.deb?29Du_evTnLbRmCS3-jWud916eoeewvUwKJ9pMmcABuUV3YodKnMQJw0tgIvtpcAx1gnwszvqbK6hfHp48AqSC2yNWeVBgUcWpHa9CbMzdbI__IAIUWQmClba0uhRh9vWrqsGFVFq2A5RMhA3SkssxItE7m7cnyT-M-peDBNbFjuoKan0hwuLZmVzf78HBOCKAHzwhVZQaYBhEcbKsG_iU99q
sudo dpkg -i nccl-repo-ubuntu1604-2.6.4-ga-cuda10.0_1-1_amd64.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev