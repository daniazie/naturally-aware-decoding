reranker_type=(
    "ratios"
    "likelihood"
    "comet"
    "combined"
    "none"
)

models=(
    "Qwen/Qwen3-4B"
    "google/gemma-3-4b-it"
)

for reranker in ${reranker_type[@]}; do
    for model in ${models[@]}; do
        bash scripts/decode.sh ${model} ${reranker}
    done
done