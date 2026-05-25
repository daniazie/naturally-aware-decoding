reranker_type=(
    "none"
    "ratios"
    "self"
)

models=(
    "Qwen/Qwen3-4B"
    "google/gemma-3-4b-it"
    "google/translategemma-4b-it"
)

tgt_langs=(
    "msa"
    "kor"
    "zho"
)

datasets=(
    "NTREX/NTREX-128"
    "openlanguagedata/flores_plus"
)

for data_path in ${datasets[@]}; do
    for tgt_lang in ${tgt_langs[@]}; do
        for reranker in ${reranker_type[@]}; do
            for model in ${models[@]}; do
                bash scripts/decode.sh ${model} ${tgt_lang} ${data_path} ${reranker}
            done
        done
    done
done
bash scripts/eval.sh