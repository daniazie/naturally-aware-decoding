train_data_dir=$1
seed=$2
max_samples=$3
learning_rate=$4
epoch=$5
batch_size=$6

deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/enzh/${train_data_dir}/train.jsonl \
   --input_key messages_foreignization \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 16 \
   --max_epochs ${epoch} \
   --pretrain Qwen/Qwen2.5-0.5B \
   --save_path models/sft/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/positive \
   --save_steps -1 \
   --logging_steps 1 \
   --zero_stage 2 \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/sft/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/positive \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed}

deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/enzh/${train_data_dir}/train.jsonl \
   --input_key messages_domestication \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 16 \
   --max_epochs ${epoch} \
   --pretrain Qwen/Qwen2.5-0.5B \
   --save_path models/sft/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/negative \
   --save_steps -1 \
   --logging_steps 1 \
   --zero_stage 2 \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/sft/qwen2.5-0.5b-${train_data_dir}-${max_samples}-${seed}/negative \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed}