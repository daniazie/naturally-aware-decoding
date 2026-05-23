model=$1
reranker_type=$2

metric_types=(
    "logprobs"
    "entropy"
)

metrics=(
    "perplexity"
    "entropy"
)

granularities=(
    "segment"
    "sequence"
)

if [[ $reranker_type == "ratios" ]]; then
    for granularity in ${granularities[@]}; do 
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang kor \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enko_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 20 \
        #     --granularity ${granularity} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --vllm \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enms_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 20 \
            --granularity ${granularity} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang zho \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enzh_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 20 \
        #     --granularity ${granularity} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --vllm \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "likelihood" ]]; then
    for metric in ${metrics[@]}; do 
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang kor \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${metric}_enko_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 40 \
        #     --metric ${metric} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --vllm \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_run5/${reranker_type}/${metric}_segment_enms_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 30 \
            --temperature 1.2 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type} \
            --per_segment_eval
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang zho \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${metric}_enzh_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 40 \
        #     --metric ${metric} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --vllm \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "self" ]]; then
    for metric in ${metrics[@]}; do 
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang kor \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${metric}_enko_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 40 \
        #     --metric ${metric} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 16 \
            --output_file results/qa_decode_run5/${reranker_type}/${metric}_segment_enms_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 20 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_new_tokens 512 \
            --batch_size 16 \
            --reranker_type ${reranker_type} \
            --per_segment_eval \
            --output_logits \
            --return_dict_in_generate \
            --temperature 1.2 \
            --do_sample
        # uv run --extra cu130 python src/qa_decoding/generate.py \
        #     --model ${model} \
        #     --data_path NTREX/NTREX-128 \
        #     --tgt_lang zho \
        #     --best_of 32 \
        #     --output_file results/qa_decode_run5/${reranker_type}/${metric}_enzh_ntrex-128_results.json \
        #     --top_p 0.95 \
        #     --top_k 40 \
        #     --metric ${metric} \
        #     --return_score true \
        #     --normalise_scores true \
        #     --max_tokens 512 \
        #     --batch_size 16 \
        #     --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "combined" ]]; then
    for granularity in ${granularities[@]}; do 
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enko_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 40 \
            --granularity ${granularity} \
            --return_score true \
            --return_nat true \
            --return_comet true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enms_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 40 \
            --granularity ${granularity} \
            --return_score true \
            --return_nat true \
            --return_comet true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang zho \
            --best_of 32 \
            --output_file results/qa_decode_run5/${reranker_type}/${granularity}_enzh_ntrex-128_results.json \
            --top_p 0.95 \
            --top_k 40 \
            --granularity ${granularity} \
            --return_score true \
            --return_nat true \
            --return_comet true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "comet" ]]; then
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang kor \
        --best_of 32 \
        --output_file results/qa_decode_run5/${reranker_type}/enko_ntrex-128_results.json \
        --top_p 0.95 \
        --top_k 40 \
        --return_score true \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang msa \
        --best_of 32 \
        --output_file results/qa_decode_run5/${reranker_type}/enms_ntrex-128_results.json \
        --top_p 0.95 \
        --top_k 40 \
        --return_score true \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang zho \
        --best_of 32 \
        --output_file results/qa_decode_run5/${reranker_type}/enzh_ntrex-128_results.json \
        --top_p 0.95 \
        --top_k 40 \
        --return_score true \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
else
    # uv run --extra cu130 python src/qa_decoding/generate.py \
    #     --model ${model} \
    #     --data_path NTREX/NTREX-128 \
    #     --tgt_lang kor \
    #     --best_of 32 \
    #     --output_file results/qa_decode_run5/${reranker_type}/enko_ntrex-128_results.json \
    #     --top_p 0.95 \
    #     --top_k 40 \
    #     --max_tokens 512 \
    #     --vllm \
    #     --batch_size 16 \
    #     --reranker_type ${reranker_type}
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang msa \
        --best_of 32 \
        --output_file results/qa_decode_run5/${reranker_type}/enms_ntrex-128_results.json \
        --top_p 0.95 \
        --top_k 40 \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
    # uv run --extra cu130 python src/qa_decoding/generate.py \
    #     --model ${model} \
    #     --data_path NTREX/NTREX-128 \
    #     --tgt_lang zho \
    #     --best_of 32 \
    #     --output_file results/qa_decode_run5/${reranker_type}/enzh_ntrex-128_results.json \
    #     --top_p 0.95 \
    #     --top_k 40 \
    #     --max_tokens 512 \
    #     --vllm \
    #     --batch_size 16 \
    #     --reranker_type ${reranker_type}
fi