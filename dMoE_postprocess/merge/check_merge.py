#
# author : 加藤　純
#
import sys
import torch

input_dir = sys.argv[1]
output_dir = sys.argv[2]
mps = int(sys.argv[3])
eps = int(sys.argv[4])

sd = torch.load(output_dir + "/consolidated.00.pth", map_location="cpu")

for mp in range(mps):
    split_ckpt = {}
    for k, v in sd.items():
        if "experts.0." in k:
            all_exp = [sd[k.replace("experts.0.", f"experts.{i}.")] for i in range(eps)]
            if "w2" in k:
                all_exp = [torch.t(_) for _ in all_exp]
            k = k.replace("experts.0.", "").replace(".weight", "")
            split_ckpt[k] = torch.cat([torch.chunk(_, mps)[mp] for _ in all_exp])

    file = f'consolidated.{mp:02d}-of-{mps:02d}.model.pth'
    old_sd = torch.load(input_dir + "/" + file, map_location="cpu")["model"]
    for k, v in split_ckpt.items():
        assert torch.equal(old_sd["llma." + k], v), f'mismatch : {file}_{k}'


#
# author : 咸　毅成
# editor : 加藤　純
#
import re


old_files = [f'consolidated.{mp:02d}-of-{mps:02d}.model.pth' for mp in range(mps)]
new_file = "consolidated.00.pth"

# expert以外変換処理された層のパターン
layer_patterns = [
    "layers\.(\d+)\.attention\.wq\.weight",
    "layers\.(\d+)\.attention\.wk\.weight",
    "layers\.(\d+)\.attention\.wv\.weight",
    "layers\.(\d+)\.attention\.wo\.weight",
]

def check_weights(input_dir, output_dir, old_files, new_file):
    old_sd = [torch.load(input_dir + "/" + file, map_location="cpu")["model"] for file in old_files]
    new_sd = torch.load(output_dir + "/" +new_file, map_location="cpu")

    for key in new_sd.keys():
        if key == "tok_embeddings.weight" or re.match("layers\.(\d+)\.attention\.wo\.weight", key):
            old_value_list = [sd["llma." + key] for sd in old_sd]
            old_value = torch.cat(old_value_list, dim=1)
            new_value = new_sd[key]
            assert torch.equal(new_value, old_value), f'mismatch : {key}'

        #判定が一部重複しているが、明示的に記載している
        elif ".attention_norm.weight" in key or "norm.weight" in key or ".ffn_norm.weight" in key or ".gate.weight" in key:
            old_value = old_sd[0]["llma." + key]
            new_value = new_sd[key]
            assert torch.equal(new_value, old_value), f'mismatch : {key}'

        else:
            for pattern in layer_patterns:
                if re.match(pattern, key) or key == "output.weight":
                    old_value_list = [sd["llma." + key] for sd in old_sd]
                    old_value = torch.cat(old_value_list, dim=0)
                    new_value = new_sd[key]
                    assert torch.equal(new_value, old_value), f'mismatch : {key}'
                    break


check_weights(input_dir, output_dir, old_files, new_file)
