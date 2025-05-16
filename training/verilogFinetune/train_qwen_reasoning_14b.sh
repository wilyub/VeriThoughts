# Set environment variables
export NPROC_PER_NODE=8        # Using all 8 GPUs
export NNODES=1                # Single node
export NODE_RANK=0             # This is the first (only) node
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
export MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"  # Path to Qwen2.5 Coder model
export OUTPUT_PATH="/home/ubuntu/qwenReasoning/output/qwen2.5-14b-instruct-finetune-reasoning"
export DS_CONFIG_PATH="./ds_config_zero3.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS /home/ubuntu/LLaMA-Factory/src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --model_name_or_path $MODEL_PATH \
    --dataset reasoning_dataset \
    --template qwen \
    --finetuning_type full \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --ddp_timeout 180000000 \
    --learning_rate 8.0e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 16384 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16
