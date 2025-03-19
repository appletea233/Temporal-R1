set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags. Output the final answer in JSON format."""

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=examples/data_config/tvg.yaml \
    data.val_files=examples/data_config/tvg.yaml \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=16 \
    worker.actor.global_batch_size=16 \
    worker.actor.entropy_coeff=1e-3 \
    worker.actor.kl_loss_coef=1e-2 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_tvg \
    trainer.n_gpus_per_node=8 \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=50 \
    trainer.val_before_train=false \
    trainer.logger=[\"console\",\"wandb\"] \
    data.min_pixels=3136 \
    data.max_pixels=1605632 \
    data.system_prompt="${SYSTEM_PROMPT}" 

