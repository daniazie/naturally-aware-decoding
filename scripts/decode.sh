model=$1
tgt_lang=$2
data_path=$3
reranker_type=$4

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
        uv run src/qa_decoding/generate.py \
            --model ${model} \
            --data_path ${data_path} \
            --tgt_lang ${tgt_lang} \
            --best_of 32 \
            --output_file results/qa_decode_run/${reranker_type}/${granularity}_${tgt_lang}_${data_path}_results.json \
            --granularity ${granularity} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 64 \
            --reranker_type ${reranker_type} \
            
    done
elif [[ $reranker_type == "likelihood" ]]; then
    for metric in ${metrics[@]}; do 
        uv run src/qa_decoding/generate.py \
            --model ${model} \
            --data_path ${data_path} \
            --tgt_lang ${tgt_lang} \
            --best_of 32 \
            --output_file results/qa_decode_run/${reranker_type}/${metric}_segment_${tgt_lang}_${data_path}_results.json \
            --top_p 0.95 \
            --top_k 30 \
            --temperature 0.9 \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_tokens 512 \
            --vllm \
            --batch_size 64 \
            --reranker_type ${reranker_type} \
            --per_segment_eval \
            
    done
elif [[ $reranker_type == "self" ]]; then
    for metric in ${metrics[@]}; do 
        uv run src/qa_decoding/generate.py \
            --model ${model} \
            --data_path ${data_path} \
            --tgt_lang ${tgt_lang} \
            --best_of 32 \
            --output_file results/qa_decode_run/${reranker_type}/${metric}_segment_${tgt_lang}_${data_path}_results.json \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_new_tokens 512 \
            --batch_size 64 \
            --reranker_type ${reranker_type} \
            --per_segment_eval \
            --output_logits \
            --return_dict_in_generate \
            --do_sample \
            
        uv run src/qa_decoding/generate.py \
            --model ${model} \
            --data_path ${data_path} \
            --tgt_lang ${tgt_lang} \
            --best_of 32 \
            --output_file results/qa_decode_run/${reranker_type}/${metric}_segment_${tgt_lang}_${data_path}_results.json \
            --metric ${metric} \
            --return_score true \
            --normalise_scores true \
            --max_new_tokens 512 \
            --batch_size 64 \
            --reranker_type ${reranker_type} \
            --per_segment_eval \
            --output_logits \
            --return_dict_in_generate \
            --do_sample \
            --per_segment_eval \
            
    done
else
    uv run src/qa_decoding/generate.py \
        --model ${model} \
        --data_path ${data_path} \
        --tgt_lang ${tgt_lang} \
        --best_of 32 \
        --output_file results/qa_decode_run/${reranker_type}/${tgt_lang}_${data_path}_results.json \
        --max_tokens 512 \
        --vllm \
        --batch_size 64 \
        --reranker_type ${reranker_type}
fi