reranker_type=(
    "natural"
    "comet"
    "combined"
    "none"
)

for reranker in ${reranker_type[@]}; do
    bash scripts/decode.sh ${reranker}
done