learning_rate=$1
epoch=$2
batch_size=$3

uv run --extra cu130 deepspeed --master_port 29501 --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/train_multilingual.jsonl \
   --input_key messages_foreignization \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 32 \
   --max_epochs ${epoch} \
   --pretrain Qwen/Qwen3-0.6B \
   --save_path models/sft/qwen2.5-0.5b-multilingual/positive \
   --save_steps -1 \
   --logging_steps 1 \
   --zero_stage 2 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/sft/qwen2.5-0.5b-multilingual/positive \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed 42

uv run --extra cu130 deepspeed --master_port 29501 --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/synthetic/train_multilingual.jsonl \
   --input_key messages_domestication \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 32 \
   --max_epochs ${epoch} \
   --pretrain Qwen/Qwen3-0.6B \
   --save_path models/sft/qwen2.5-0.5b-multilingual/negative \
   --save_steps -1 \
   --logging_steps 1 \
   --zero_stage 2 \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/sft/qwen2.5-0.5b-multilingual/negative \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed 42