train_data_dir=$1
seed=$2
max_samples=$3

deepspeed --module openrlhf.cli.train_rm \
    --max_len 1024 \
    --dataset data/synthetic/enzh/${train_data_dir}/train.jsonl \
    --chosen_key messages_foreignization \
    --rejected_key messages_domestication \
    --apply_chat_template \
    --train_batch_size 16 \
    --micro_train_batch_size 16 \
    --max_epochs 3 \
    --pretrain Qwen/Qwen2.5-0.5B \
    --save_path models/rm_contrast/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/positive \
    --save_steps -1 \
    --logging_steps 1 \
    --zero_stage 2 \
    --max_samples ${max_samples} \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --use_tensorboard logs/rm_contrast/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/positive \
    --learning_rate 4e-6 \
    --l2 0.05 \
    --lr_warmup_ratio 0.1 \
    --seed ${seed}

deepspeed --module openrlhf.cli.train_rm \
    --max_len 1024 \
    --dataset data/synthetic/enzh/${train_data_dir}/train.jsonl \
    --chosen_key messages_domestication \
    --rejected_key messages_foreignization \
    --apply_chat_template \
    --train_batch_size 16 \
    --micro_train_batch_size 16 \
    --max_epochs 3 \
    --pretrain Qwen/Qwen2.5-0.5B \
    --save_path models/rm_contrast/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/negative \
    --save_steps -1 \
    --logging_steps 1 \
    --zero_stage 2 \
    --max_samples ${max_samples} \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --use_tensorboard logs/rm_contrast/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/negative \
    --learning_rate 4e-6 \
    --l2 0.05 \
    --lr_warmup_ratio 0.1 \
    --seed ${seed}