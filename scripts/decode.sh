model=$1
reranker_type=$2
granularities=(
    "token"
    "sequence"
    "segment"
)

metrics=(
    "perplexity"
    "entropy"
    "surprisal"
    "logprobs"
)

if [[ $reranker_type == "ratios" ]]; then
    for granularity in ${granularities[@]}; do 
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enko_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enms_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang zho \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enzh_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "likelihood" ]]; then
    for metric in ${metrics[@]}; do 
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${metric}_enko_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${metric}_enms_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang zho \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${metric}_enzh_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 16 \
            --reranker_type ${reranker_type}
    done
elif [[ $reranker_type == "combined" ]]; then
    for granularity in ${granularities[@]}; do 
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model ${model} \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enko_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
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
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enms_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
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
            --output_file results/qa_decode_run3/${reranker_type}_${granularity}_enzh_ntrex-128_results.json \
            --temperature 0.9 \
            --top_k 20 \
            --top_p 0.90 \
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
        --output_file results/qa_decode_run3/${reranker_type}_enko_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
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
        --output_file results/qa_decode_run3/${reranker_type}_enms_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
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
        --output_file results/qa_decode_run3/${reranker_type}_enzh_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
        --return_score true \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
else
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang kor \
        --best_of 32 \
        --output_file results/qa_decode_run3/${reranker_type}_enko_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang msa \
        --best_of 32 \
        --output_file results/qa_decode_run3/${reranker_type}_enms_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
    uv run --extra cu130 python src/qa_decoding/generate.py \
        --model ${model} \
        --data_path NTREX/NTREX-128 \
        --tgt_lang zho \
        --best_of 32 \
        --output_file results/qa_decode_run3/${reranker_type}_enzh_ntrex-128_results.json \
        --temperature 0.9 \
        --top_k 20 \
        --top_p 0.90 \
        --max_tokens 512 \
        --vllm \
        --batch_size 16 \
        --reranker_type ${reranker_type}
fi