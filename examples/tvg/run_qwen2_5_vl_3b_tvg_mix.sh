set -x

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export DECORD_EOF_RETRY_MAX=2048001
export NCCL_NET=Socket

export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"

export VLLM_ATTENTION_BACKEND=XFORMERS
export https_proxy=http://10.70.11.196:8412
export WANDB_API_KEY=ad629497d0d035e595f6a628c1f579e7af69703c

###### for remote env setting ######
export LD_LIBRARY_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/miniconda3/bin:$LD_LIBRARY_PATH
ls /opt/rh
. /opt/rh/devtoolset-8/enable

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/conda/bin/activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/webagent_envs/easyr1-v2

GPUS=$1
SAMPLE_N=8
KL_LOSS_COEF=1e-2
GLOBAL_BS=32
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. Output the final answer in JSON format."""

MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/lihongyu/Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/video_llm/codes/VLM-R1/src/open-r1-multimodal-new/data_config/tvg_mix.yaml \
    data.val_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/video_llm/codes/VLM-R1/src/EasyR1/scripts/tvg.yaml \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.rollout_batch_size=$GLOBAL_BS \
    worker.actor.global_batch_size=$GLOBAL_BS \
    worker.actor.entropy_coeff=1e-3 \
    worker.actor.kl_loss_coef=${KL_LOSS_COEF} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=${SAMPLE_N} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_tvg_gpu_${GPUS}_v3_kl_${KL_LOSS_COEF}_n_${SAMPLE_N}_format_1_gbs_${GLOBAL_BS}_mix_large \
    trainer.n_gpus_per_node=$GPUS \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=50 \
    trainer.val_before_train=false \
    trainer.logger=[\"console\",\"wandb\"] \
    data.min_pixels=2592 \
    data.max_pixels=1605632 \
    data.system_prompt="${SYSTEM_PROMPT}" 

