reranker_type=$1
granularities=(
    "segment"
    "token"
    "sequence"
)

if [[ $reranker_type != "comet" && $reranker_type != "none" ]]; then
    for granularity in ${granularities[@]}; do 
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_${granularity}_enko_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --lang kor \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_${granularity}_enms_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --lang msa \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang zho \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_${granularity}_enzh_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --granularity ${granularity} \
            --return_score true \
            --lang zho \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
    done
else
    uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang kor \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_enko_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --return_score true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang msa \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_enms_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --return_score true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
        uv run --extra cu130 python src/qa_decoding/generate.py \
            --model Qwen/Qwen3-4B \
            --data_path NTREX/NTREX-128 \
            --tgt_lang zho \
            --best_of 32 \
            --output_file results/qa_decode_v2/${reranker_type}_enzh_ntrex-128_results.json \
            --temperature 0.7 \
            --top_k 20 \
            --top_p 0.90 \
            --return_score true \
            --max_tokens 512 \
            --vllm \
            --batch_size 8 \
            --reranker_type ${reranker_type}
fi