for seed in 10 20 30; do
    bash scripts/rm.sh oliver_twist_qwen ${seed} 1000 1e-6 3 64 per_sequence_loss
    bash scripts/rm.sh oliver_twist_qwen ${seed} 1000 1e-6 3 64 per_token_loss sequence_level
    bash scripts/rm.sh oliver_twist_qwen ${seed} 1000 1e-6 3 64 per_token_loss token_level
done

bash scripts/rm.sh coca_blog_llama 10 1000 1e-6 3 64 per_sequence_loss
bash scripts/rm.sh mixture 10 5000 2.7e-5 1 64 per_sequence_loss
bash scripts/rm.sh mixture 10 3000 2.7e-5 1 64 per_sequence_loss
bash scripts/rm.sh oliver_twist_qwen 10 1000 1e-6 3 64 per_sequence_loss

bash scripts/rm.sh coca_blog_llama 10 1000 1e-6 3 64 per_token_loss sequence_level
bash scripts/rm.sh mixture 10 5000 2.7e-5 1 64 per_token_loss sequence_level
bash scripts/rm.sh mixture 10 3000 2.7e-5 1 64 per_token_loss sequence_level
bash scripts/rm.sh oliver_twist_qwen 10 1000 1e-6 3 64 per_token_loss sequence_level

bash scripts/rm.sh coca_blog_llama 10 1000 1e-6 3 64 per_token_loss token_level
bash scripts/rm.sh mixture 10 5000 2.7e-5 1 64 per_token_loss token_level
bash scripts/rm.sh mixture 10 3000 2.7e-5 1 64 per_token_loss token_level
bash scripts/rm.sh oliver_twist_qwen 10 1000 1e-6 3 64 per_token_loss token_level