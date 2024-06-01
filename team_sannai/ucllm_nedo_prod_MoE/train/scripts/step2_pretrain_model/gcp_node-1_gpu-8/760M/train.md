
## GPUのattach

```
srun --nodelist=mlpre-g2-ghpc-17 --gpus-per-node=8 --time=06:00:00 --pty bash -i
```

```
srun --nodelist=mlpre-g2-ghpc-17 --gpus-per-node=8 --time=06:00:00 --pty bash -i
```



ノードは被らない方が良い？？
使ってなさそうなものを選ぶ


gpuがattachされてればOK  
(以下は4つの例)

```
$ nvidia-smi
Thu Apr  4 06:38:08 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA L4                      On  | 00000000:00:03.0 Off |                    0 |
| N/A   61C    P8              13W /  72W |     83MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA L4                      On  | 00000000:00:04.0 Off |                    0 |
| N/A   62C    P8              15W /  72W |     13MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA L4                      On  | 00000000:00:05.0 Off |                    0 |
| N/A   56C    P8              13W /  72W |     13MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA L4                      On  | 00000000:00:06.0 Off |                    0 |
| N/A   58C    P8              13W /  72W |     13MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2257      G   /usr/lib/xorg/Xorg                           59MiB |
|    0   N/A  N/A      2841      G   /usr/bin/gnome-shell                         10MiB |
|    1   N/A  N/A      2257      G   /usr/lib/xorg/Xorg                            4MiB |
|    2   N/A  N/A      2257      G   /usr/lib/xorg/Xorg                            4MiB |
|    3   N/A  N/A      2257      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

## train llama2

```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--output_model_dir "/persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2"
```