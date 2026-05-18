train_data_dir=$1
seed=$2
max_samples=$3
learning_rate=$4
epoch=$5
batch_size=$6
loss_type=$7

OUTPUT_DIR=models/rm/qwen2.5-0.5b-${train_data_dir}-${seed}-${loss_type}

if [[ -n "$8" ]]; then
    OUTPUT_DIR=${OUTPUT_DIR}-{$8}
    loss_level=$8
else
    loss_level=sequence_level
fi

uv run --extra cu130 src/train_rm.py \
    --config_file configs/rm_config.yaml \
    --chosen_key messages_foreignization \
    --rejected_key messages_domestication \
    --data_path t_index_reproduce/data/synthetic/enzh/${train_data_dir} \
    --max_samples ${max_samples} \
    --output_dir ${OUTPUT_DIR}/positive \
    --train_config configs/seq_training_args.yaml \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${batch_size} \
    --seed ${seed} \
    --loss_type ${loss_type} \
    --per_sequence_loss_level ${loss_level}

uv run --extra cu130 src/train_rm.py \
    --config_file configs/rm_config.yaml \
    --chosen_key messages_domestication \
    --rejected_key messages_foreignization \
    --data_path t_index_reproduce/data/synthetic/enzh/${train_data_dir} \
    --max_samples ${max_samples} \
    --output_dir ${OUTPUT_DIR}/negative \
    --train_config configs/seq_training_args.yaml \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${batch_size} \
    --seed ${seed} \
    --loss_type ${loss_type} \
    --per_sequence_loss_level ${loss_level}

