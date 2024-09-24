MODEL="meta-llama/CodeLlama-34b-Instruct-hf"
SANITIZED_MODEL_NAME=$(echo $MODEL | tr '/' '-')
DATASET="GindaChen/ultrachat-1k-chatml"
OUTPUT_DIR="${SANITIZED_MODEL_NAME}-sft-lora"
CONFIG_FILE="configs/fsdp_config.yaml"

echo "Using config file: $CONFIG_FILE"
echo " ---------------------------------------- "
cat $CONFIG_FILE
echo " ---------------------------------------- "


set -x

accelerate launch --config_file $CONFIG_FILE  train.py \
--seed 100 \
--model_name_or_path ${MODEL} \
--dataset_name ${DATASET} \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train_sft,test_sft" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 2e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 0.0 \
--warmup_ratio 0.1 \
--max_grad_norm 1.0 \
--output_dir ${OUTPUT_DIR} \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn False