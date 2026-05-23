reranker_type=(
    "ratios"
    "none"
)

models=(
    "google/gemma-3-4b-it"
    "google/translategemma-4b-it"
    "Qwen/Qwen3-4B"
)

for reranker in ${reranker_type[@]}; do
    for model in ${models[@]}; do
        bash scripts/decode.sh ${model} ${reranker}
    done
done