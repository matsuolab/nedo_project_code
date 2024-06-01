num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=3
num_gpus=$((${num_gpus_pernode} * ${num_node}))

echo ${num_gpus_pernode}
echo ${num_node}
echo ${num_gpus}