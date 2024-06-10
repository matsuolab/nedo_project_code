import argparse
import gzip
from loaders import *
import os
import json
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")

def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to download", default="refinedweb")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")
    parser.add_argument("--streaming", type=bool, default=True, help="Streaming mode")
    parser.add_argument("--index_from", type=int, default=0, help="Index to start downloading")
    parser.add_argument("--index_to", type=int, default=10000000000, help="Index to stop downloading")

    return parser.parse_args()


def main():
    args = parse_args()
    streaming = args.streaming
    split = args.split
    
    # こちらにデータセットを追加していく
    if args.dataset == "refinedweb":
        loader = refinedweb_en_loader(streaming=streaming, split=split)
        output_dir_path = os.path.join(OUTPUT_PATH, "refinedweb")
        output_path = os.path.join(output_dir_path, "refinedweb.jsonl.gz")
        content = "content"

    elif args.dataset == "slimpajama":
        loader = slimpajama_en_loader(streaming=streaming, split=split)
        output_dir_path = os.path.join(OUTPUT_PATH, "slimpajama")
        output_path = os.path.join(output_dir_path, "slimpajama.jsonl.gz")
        content = "text"

    elif args.dataset == "mc4-ja": 
        loader = mc4_ja_part_loader(streaming=streaming, split=split)
        output_dir_path = os.path.join(OUTPUT_PATH, "mc4_ja")
        output_path = os.path.join(output_dir_path, "mc4_ja.jsonl.gz")
        content = "text"
    
    elif args.dataset == "wiki-ja": 
        loader = wiki_ja_loader(streaming=streaming, split=split)
        output_dir_path = os.path.join(OUTPUT_PATH, "wiki_ja")
        output_path = os.path.join(output_dir_path, "wiki_ja.jsonl.gz")
        content = "text"

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(os.path.join(output_dir_path, "index.txt"), "a") as index_file:
        with gzip.open(output_path, "at") as f:
            if streaming == True:
                text_list = iter(loader)
                i = -1
                while True:
                    i+=1
                    if i < args.index_from: continue
                    elif args.index_to < i: break
                    else:
                        try:
                            text = next(text_list)
                            out_text = json.dumps({"text": text[content]}, ensure_ascii=False)
                            f.write(out_text+"\n")
                            index_file.write(str(i)+"\n")

                        except StopIteration:
                            break
            else:
                texts = loader[content][args.index_from:args.index_to]
                for text in texts:
                    out_text = json.dumps({"text": text}, ensure_ascii=False)
                    f.write(out_text+"\n")
 
if __name__ == "__main__":
    main()
