GPUS=$1
MODEL_PATH=$2
TASKS=$3
VISUAL_TOKEN=1605632

###### for huggingface setting ######
export HF_HOME="<Path to HF cache>" 
export TRANSFORMERS_CACHE=$HF_HOME/transformers
###### for huggingface setting ######

export DECORD_EOF_RETRY_MAX=20480

accelerate launch --num_processes=$GPUS --main_process_port 12345 -m lmms_eval \
  --model qwen2_5_vl_r1 \
  --model_args "pretrained=$MODEL_PATH,total_pixels=$VISUAL_TOKEN,task_type=tvg" \
  --device cuda \
  --tasks $TASKS \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix debug \
  --output_path $MODEL_PATH/lmms_eval/logs/gpu_${GPUS}_visual_token_${VISUAL_TOKEN}_r1/$TASKS \
  --verbosity=DEBUG

# temporal_grounding_charades,temporal_grounding_activitynet, 