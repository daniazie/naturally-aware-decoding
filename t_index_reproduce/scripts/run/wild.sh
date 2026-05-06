model_dir="mixture"

for nsample in 5000 3000 1000; do
    python src/t_index.py \
        --config recipes/wild.yaml \
        --data_path data/wild/pointwise.jsonl \
        --model_positive models/sft/qwen2.5-0.5b-${model_dir}-${nsample}-10/positive \
        --model_negative models/sft/qwen2.5-0.5b-${model_dir}-${nsample}-10/negative \
        --prompt_field source \
        --completion_field translation \
        --output_file results/wild_t_index_sample${nsample}.jsonl
    python src/supervised.py \
        --config recipes/wild.yaml \
        --model_path models/rm/qwen2.5-0.5b-${model_dir}-${nsample}-10 \
        --output_file results/wild_rm_samples${nsample}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/wild.yaml \
        --model_path models/dpo/qwen2.5-0.5b-${model_dir}-${nsample}-10 \
        --output_file results/wild_dpo_samples${nsample}.jsonl \
        --model_type dpo
done

model_dir="oliver_twist_qwen"
nsample=1000
python src/t_index.py \
    --config recipes/wild.yaml \
    --data_path data/wild/pointwise.jsonl \
    --model_positive models/sft/qwen2.5-0.5b-${model_dir}-${nsample}-10/positive \
    --model_negative models/sft/qwen2.5-0.5b-${model_dir}-${nsample}-10/negative \
    --prompt_field source \
    --completion_field translation \
    --output_file results/wild_t_index_sample${nsample}_onedomain.jsonl

python src/t_index.py \
    --config recipes/wild.yaml \
    --data_path data/wild/pointwise.jsonl \
    --model_positive models/sft/qwen2.5-0.5b-coca_blog_llama-${nsample}-10/positive \
    --model_negative models/sft/qwen2.5-0.5b-oliver_twist_qwen-${nsample}-10/negative \
    --prompt_field source \
    --completion_field translation \
    --output_file results/wild_t_index_sample${nsample}_unpaired.jsonl