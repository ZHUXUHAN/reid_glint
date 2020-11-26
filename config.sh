# nvidia
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_BROADCAST=NCLL
export HOROVOD_CACHE_CAPACITY=1024
export HOROVOD_FUSION_THRESHOLD=4194304



export MXNET_GPU_WORKER_NTHREADS=1
export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

PYTHON_EXEC=/home/ubuntu/miniconda3/envs/mxnet/bin/python

${PYTHON_EXEC} train_memory.py \
--dataset market_ducket_cuhk03_person28w \
--loss cosface \
--network r50 \
--models-root /home/ubuntu/zhuxuhan/reid/reid_model/market_ducket_cuhk03_person28w-r50
