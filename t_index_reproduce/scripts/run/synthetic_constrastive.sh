model_dir="oliver_twist_qwen"

for seed in 10 20 30; do
    python src/unsupervised.py \
        --config recipes/synthetic_enzh.yaml \
        --model_positive models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_enzh_unsupervised_sft_seed${seed}.jsonl \
        --model_type sft
    python src/unsupervised.py \
        --config recipes/synthetic_enzh.yaml \
        --model_positive models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_enzh_unsupervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_enzh.yaml \
        --model_path models/rm/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_enzh_supervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_enzh.yaml \
        --model_path models/dpo/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_enzh_supervised_dpo_seed${seed}.jsonl \
        --model_type dpo
    python src/supervised.py \
        --config recipes/synthetic_enzh.yaml \
        --model_path models/clf/xlm-roberta-large-${model_dir}-${seed} \
        --output_file results_v2/synthetic_enzh_supervised_clf_seed${seed}.jsonl \
        --model_type clf
done

for seed in 10 20 30; do
    python src/unsupervised.py \
        --config recipes/synthetic_ende.yaml \
        --model_positive models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_ende_unsupervised_sft_seed${seed}.jsonl \
        --model_type sft
    python src/unsupervised.py \
        --config recipes/synthetic_ende.yaml \
        --model_positive models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_ende_unsupervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_ende.yaml \
        --model_path models/rm/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_ende_supervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_ende.yaml \
        --model_path models/dpo/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_ende_supervised_dpo_seed${seed}.jsonl \
        --model_type dpo
    python src/supervised.py \
        --config recipes/synthetic_ende.yaml \
        --model_path models/clf/xlm-roberta-large-${model_dir}-${seed} \
        --output_file results_v2/synthetic_ende_supervised_clf_seed${seed}.jsonl \
        --model_type clf
done

for seed in 10 20 30; do
    python src/unsupervised.py \
        --config recipes/synthetic_enfr.yaml \
        --model_positive models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/sft/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_enfr_unsupervised_sft_seed${seed}.jsonl \
        --model_type sft
    python src/unsupervised.py \
        --config recipes/synthetic_enfr.yaml \
        --model_positive models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/positive \
        --model_negative models/rm_contrast/qwen2.5-0.5b-${model_dir}-1000-${seed}/negative \
        --output_file results_v2/synthetic_enfr_unsupervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_enfr.yaml \
        --model_path models/rm/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_enfr_supervised_rm_seed${seed}.jsonl \
        --model_type rm
    python src/supervised.py \
        --config recipes/synthetic_enfr.yaml \
        --model_path models/dpo/qwen2.5-0.5b-${model_dir}-1000-${seed} \
        --output_file results_v2/synthetic_enfr_supervised_dpo_seed${seed}.jsonl \
        --model_type dpo
    python src/supervised.py \
        --config recipes/synthetic_enfr.yaml \
        --model_path models/clf/xlm-roberta-large-${model_dir}-${seed} \
        --output_file results_v2/synthetic_enfr_supervised_clf_seed${seed}.jsonl \
        --model_type clf
done