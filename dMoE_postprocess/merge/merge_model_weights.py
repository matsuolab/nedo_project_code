import json
import re
import argparse
import torch

pattern_to_merge_dim = {
        "^tok_embeddings.weight$" : 1,
        "^layers.(\d+).attention.wq.weight$" : 0,
        "^layers.(\d+).attention.wq.bias$" : 0,
        "^layers.(\d+).attention.wk.weight$" : 0,
        "^layers.(\d+).attention.wk.bias$" : 0,
        "^layers.(\d+).attention.wv.weight$" : 0,
        "^layers.(\d+).attention.wv.bias$" : 0,        
        "^layers.(\d+).attention.wo.weight$" : 1,
        "^output.weight$" : 0,
        "^output.bias$" : 0,
    }


def merge_sparse(input_dir, output_dir, mps, eps):
    merged_weights = {}

    for mp in range(mps):
        part_weights = torch.load(f'{input_dir}/consolidated.{mp:02d}-of-{mps:02d}.model.pth', map_location="cpu")['model']

        for key, value in part_weights.items():
            key = key.replace("llma.","")
            if "feed_forward.w" in key:
                eps_value = torch.chunk(value,eps)
                for ep in range(eps):
                    new_key = key.replace("feed_forward.w",f'feed_forward.experts.{ep}.w')+".weight"
                    if new_key not in merged_weights:
                        merged_weights[new_key] = eps_value[ep]
                    else:
                        merged_weights[new_key] = torch.cat([merged_weights[new_key], eps_value[ep]], dim=0)
            else:
                if key not in merged_weights:
                    merged_weights[key] = value
                else:
                    for pattern, merge_dim in pattern_to_merge_dim.items():
                        if re.compile(pattern).match(key):
                            merged_weights[key] = torch.cat([merged_weights[key], value], dim=merge_dim)
                            break

        print(f'|--OK : model_parallel:{mp}')

    for k,v in merged_weights.items():
        if "w2" in k:
            merged_weights[k] = torch.t(v)

    torch.save(merged_weights, f'{output_dir}/consolidated.00.pth')
    print(f'|--OK : consolidated.00.pth saved')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mps", type=int, required=True)
    parser.add_argument("--eps", type=int, required=True)
    args = parser.parse_args()
    print(f"|--{args = }")
    return args


def main() -> None:
    args = parse_arguments()
    merge_sparse(args.input_dir, args.output_dir, args.mps, args.eps)

if __name__ == "__main__":
    main()
