cd t_index_reproduce

for seed in 10 20 30; do
    bash scripts/train/sft.sh oliver_twist_qwen ${seed} 1000 1e-6 3 16
    bash scripts/train/dpo.sh oliver_twist_qwen ${seed} 1000 16
    bash scripts/train/rm.sh oliver_twist_qwen ${seed} 1000
    bash scripts/train/xlmr.sh oliver_twist_qwen ${seed}
done

bash scripts/train/sft.sh coca_blog_llama 10 1000 1e-6 3 16

bash scripts/train/sft.sh mixture 10 5000 2.7e-5 1 32
bash scripts/train/sft.sh mixture 10 3000 2.7e-5 1 32

bash scripts/train/sft.sh mixture 10 1000 1e-6 3 16

for n_samples in 5000 3000 1000; do
    bash scripts/train/rm.sh mixture 10 ${n_samples}
done

for n_samples in 5000 3000 1000; do
    bash scripts/train/dpo.sh mixture 10 ${n_samples} 8
done

bash scripts/run/synthtic.sh
bash scripts/run/wild.sh

cd ..