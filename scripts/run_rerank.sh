uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang zho \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_sequence_enzh_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity segment \
    --return_score true \
    --lang zho \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    --segment_level \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang zho \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_token_enzh_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity token \
    --return_score true \
    --lang zho \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang zho \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_sequence_enzh_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity sequence \
    --return_score true \
    --lang zho \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang kor \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_token_enko_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity segment \
    --return_score true \
    --lang kor \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    --segment_level \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang kor \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_token_enko_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity token \
    --return_score true \
    --lang kor \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang kor \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_sequence_enko_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity sequence \
    --return_score true \
    --lang kor \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang msa \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_token_enms_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity segment \
    --return_score true \
    --lang msa \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    --segment_level \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang msa \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_token_enms_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity token \
    --return_score true \
    --lang msa \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1

uv run --extra cu130 python src/qa_decoding/generate.py \
    --model Qwen/Qwen3.5-9B \
    --data_path NTREX/NTREX-128 \
    --tgt_lang msa \
    --best_of 32 \
    --output_file results/seg_qa_decode/t-index_sequence_enms_ntrex-128_results.json \
    --temperature 0.6 \
    --top_k 20 \
    --top_p 0.95 \
    --granularity sequence \
    --return_score true \
    --lang msa \
    --normalise_scores false \
    --max_tokens 512 \
    --vllm \
    --batch_size 16 \
    # --num_beams 1
