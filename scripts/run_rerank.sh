reranker_type=(
    "none"
    "natural"
    "comet"
    "combined"
)

for reranker in ${reranker_type[@]}; do
    bash scripts/decode.sh ${reranker}
done