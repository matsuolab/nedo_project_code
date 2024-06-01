import os
import random
import json
import csv
from pathlib import Path
import concurrent.futures
import time
import math
from tqdm import tqdm


def get_file_info(dataset_path, extensions):
    file_info = []
    total_size = 0
    if not os.path.exists(dataset_path):
        print("not found!!", dataset_path)
    for extension in extensions:
        for file_path in Path(dataset_path).rglob(f"*.{extension}"):
            file_size = os.path.getsize(file_path)
            file_info.append((str(file_path), file_size, file_path.name))
            total_size += file_size
    return file_info, total_size

def get_file_paths(dataset_path, extensions):
    file_paths = []
    for extension in extensions:
        file_paths.extend(list(Path(dataset_path).rglob(f"*.{extension}")))
    print(f"file_paths: {file_paths}")
    return file_paths

def get_random_files(file_paths, size_limit):
    random.shuffle(file_paths)
    total_size = 0
    selected_files = []
    for file_path in file_paths:
        file_size = os.path.getsize(file_path)
        if total_size + file_size <= size_limit:
            selected_files.append(file_path)
            total_size += file_size
    print(f"selected_files: {selected_files}")
    print(f"total_size: {total_size}")
    return selected_files


def __read_jsonl_file(file_path):
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            item = json.loads(line)
            if isinstance(item, dict) and 'text' in item:
                data.append(item['text'])
            else:
                data.append(item)
    return data

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()


def read_jsonl_files(file_paths):
    data = []
    file_item_counts = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_items = []
            for line in f:
                item = json.loads(line)
                if 'text' in item:
                    file_items.append(item['text'])
                else:
                    file_items.append(item)
            data.extend(file_items)
            file_item_counts[str(file_path)] = len(file_items)
    return data, file_item_counts


def write_file(file_path, data, output_extension):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    output_file = f"{file_path}.{output_extension}"
    if output_extension == 'jsonl':
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(output_file, 'w') as f:
            f.write('\n'.join(data))



def write_selected_files(file_path, selected_files):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("dataset_file_path,count\n")
        for selected_file in selected_files:
            f.write(f"{selected_file[0]},{selected_file[1]}\n")

def write_selected_file_counts(file_path, selected_file_counts):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("dataset_file_path,count\n")
        for file_path, count in selected_file_counts.items():
            f.write(f"{file_path},{count}\n")


def split_file(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension, selected_files_dir):
    print("Executing split_file method")
    file_paths = get_file_paths(dataset_path, extensions)
    size_bytes = size * 1024 * 1024 * 1024  # Convert size to bytes

    file_info, total_size = get_file_info(dataset_path, extensions)

    selected_files = []
    selected_data = []
    output_data = [] # この行を追加
    current_size = 0

    # サイズ制限以下のファイルを選択
    for file_path, file_size, _ in file_info:
        if file_size <= size_bytes:
            selected_files.append(file_path)

    # サイズ制限以下のファイルがない場合は、サイズ制限より大きいファイルからランダムに1つ選択
    if not selected_files:
        large_files = [file_path for file_path, file_size, _ in file_info if file_size > size_bytes]
        if not large_files:
            print(f"No files found larger than {size} GB in {dataset_path}")
            return
        selected_file = random.choice(large_files)
        selected_files = [selected_file]

    print(f"selected_files:\n{selected_files}")

    corpus_name_str = '_'.join(corpus_names)
    selected_files_dir = os.path.join(selected_files_dir, os.path.basename(dataset_path), '_'.join(corpus_names), method)
    os.makedirs(selected_files_dir, exist_ok=True)
    pickup_file_name = f"selected_files_{os.path.basename(dataset_path)}_{corpus_name_str}_{method}_{size}.csv"
    pickup_file_path = os.path.join(selected_files_dir, pickup_file_name)
    write_selected_files(pickup_file_path, [(str(path), 0) for path in selected_files])

    # 選択したファイルからデータを読み込み
    if 'jsonl' in extensions:
        selected_data = read_jsonl_files(selected_files)[0]
    elif 'txt' in extensions:
        selected_data = read_txt_files(selected_files)

    random.shuffle(selected_data)

    for item in selected_data:
        item_size = len(str(item).encode('utf-8'))
        if current_size + item_size <= size_bytes:
            output_data.append(item)
            current_size += item_size
        else:
            break

    output_file = f"{dst_dataset_path}/{os.path.basename(dataset_path)}_{corpus_name_str}_{method}_{size}.{output_extension}"
    write_file(output_file, output_data, output_extension)



def process_data(data, file_size, corpus_names, dst_dataset_path, method, size, output_extension, dataset_path):
    current_size = sum(len(str(item).encode('utf-8')) for item in data)
    print(f"Processing data, initial size: {current_size} bytes")
    output_data = [{"text": str(item)} for item in data]

    while current_size > file_size:
        if not output_data:
            print("No more items to remove from output_data")
            break
        item = output_data.pop()
        item_text = item.get('text', json.dumps(item, ensure_ascii=False)) if isinstance(item, dict) else str(item)
        item_size = len(item_text.encode('utf-8'))
        data.insert(0, item)
        current_size -= item_size
        print(f"Removed item from data, new size: {current_size} bytes")

    corpus_name_str = '_'.join(corpus_names)
    output_file = f"{dst_dataset_path}/{os.path.basename(dataset_path)}_{corpus_name_str}_{method}_{size}.{output_extension}"
    print(f"Writing output file: {output_file}")
    write_file(output_file, output_data, output_extension)
    print(f"Wrote output file: {output_file}, size: {current_size / (1024 * 1024 * 1024):.2f} GB")
    return output_file

def split_all(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension):
    print("Executing split_all method")
    file_paths = get_file_paths(dataset_path, extensions)
    size_bytes = size * 1024 * 1024 * 1024  # Convert size to bytes
    print(f"Target size in GB: {size}")

    data = []
    if 'jsonl' in extensions:
        data, _ = read_jsonl_files(file_paths)
        print(f"Total data items loaded from JSONL files: {len(data)}")
    elif 'txt' in extensions:
        data = read_txt_files(file_paths)
        print(f"Total data items loaded from TXT files: {len(data)}")

    random.shuffle(data)

    output_file = process_data(data, size_bytes, corpus_names, dst_dataset_path, method, size, output_extension, dataset_path)
    print(f"Data processed, output file: {output_file}")

    print("Processing completed")


def process_file(file_path, file_size, total_size, size_bytes, extensions):
    print(f"  Processing started for file: {file_path}")
    start_time = time.time()

    p_i = file_size / total_size
    output_i_bytes = int(p_i * size_bytes)
    print(f"    File probability: {p_i:.4f}")
    print(f"    Target file size: {output_i_bytes} bytes")

    if 'jsonl' in extensions:
        data = __read_jsonl_file(file_path)
    elif 'txt' in extensions:
        data = read_txt_file(file_path)

    selected_data = []
    current_size = 0

    for item in random.sample(data, len(data)):
        if isinstance(item, dict) and 'text' in item:
            item_text = item['text']
        else:
            item_text = item

        item_size = len(str(item_text).encode('utf-8'))
        if current_size + item_size <= output_i_bytes:
            selected_data.append({"text": item_text})
            current_size += item_size
        if current_size >= output_i_bytes:
            break


    print(f"    Selected items: {len(selected_data)}")
    print(f"    Selected size: {current_size} bytes")

    end_time = time.time()
    print(f"  Processing completed for file: {file_path}")
    print(f"  Time taken: {end_time - start_time:.2f} seconds")

    return file_path, selected_data, current_size

def split_all_with_count(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension, selected_files_dir):
    print("Executing split_all_with_count method")
    start_time = time.time()

    file_info, total_size = get_file_info(dataset_path, extensions)
    size_bytes = size * 1024 * 1024 * 1024  # Convert size to bytes

    print(f"Total files: {len(file_info)}")
    print(f"Total size: {total_size} bytes")
    print(f"Target size: {size_bytes} bytes")

    corpus_name_str = '_'.join(corpus_names)
    if not os.path.exists(dst_dataset_path):
        print("created...", dst_dataset_path)
        os.makedirs(dst_dataset_path)
    output_file = f"{dst_dataset_path}/{os.path.basename(dataset_path)}_{corpus_name_str}_{method}_{size}.{output_extension}"

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_path, file_size, _ in file_info:
            future = executor.submit(process_file, file_path, file_size, total_size, size_bytes, extensions)
            futures.append(future)

        selected_files = {}
        with open(output_file, 'w') as outfile:
            for future in concurrent.futures.as_completed(futures):
                file_path, selected_data, file_size = future.result()
                selected_files[file_path] = file_size

                for item in tqdm(selected_data):
                    # print(f"Item type: {type(item)}")
                    # print(f"Item: {item}")
                    if output_extension == 'jsonl':
                        json_data = json.dumps(item, ensure_ascii=False)
                        # print(f"JSON data: {json_data}")
                        outfile.write(json_data + '\n')
                    else:
                        outfile.write(str(item) + '\n')


    print(f"Output file: {output_file}")
    total_selected_items = 0
    with open(output_file, 'r') as infile:
        for line_num, line in enumerate(infile, start=1):
            try:
                item = json.loads(line)
                total_selected_items += len(item)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError at line {line_num}: {e}")
                print(f"Skipping line: {line.strip()}")

    print(f"Total selected items: {total_selected_items}")
    print(f"Total selected size: {sum(selected_files.values())} bytes")

    print(f"Total selected size: {sum(selected_files.values())} bytes")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    selected_files_dir = os.path.join(selected_files_dir, os.path.basename(dataset_path), '_'.join(corpus_names), 'split_all_with_count')
    os.makedirs(selected_files_dir, exist_ok=True)
    pickup_file_name = f"selected_files_{os.path.basename(dataset_path)}_{corpus_name_str}_{method}_{size}.csv"
    pickup_file_path = os.path.join(selected_files_dir, pickup_file_name)
    write_selected_file_counts(pickup_file_path, selected_files)
    print(f"Selected files written to: {pickup_file_path}")



def write_selected_file_counts(file_path, selected_file_counts):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("dataset_file_path,count\n")
        for file_path, count in selected_file_counts.items():
            f.write(f"{file_path},{count}\n")


def main():
    with open('dataset_split_settings.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_path = os.path.abspath(row['dataset_path'])
            print(f"dataset_path: {dataset_path}")
            corpus_names = row['corpus_name'].split(',')
            print(f"corpus_names: {corpus_names}")
            size = float(row['size'])
            print(f"size: {size}")
            dst_dataset_path = os.path.abspath(row['dst_dataset_path'])
            print(f"dst_dataset_path: {dst_dataset_path}")
            method = row['method']
            print(f"method: {method}")
            extensions = row['extension'].split(',')
            print(f"extensions: {extensions}")
            output_extension = row['output_extension']
            print(f"output_extension: {output_extension}")
            selected_files_dir = os.path.abspath(row.get('selected_files_dir', './selected_files'))
            print(f"selected_files_dir: {selected_files_dir}")
            os.makedirs(selected_files_dir, exist_ok=True)
            skip = row['skip']
            if skip is not None and int(skip) == 1:
                print("skip...", dataset_path)
            if method == 'split_file':
                split_file(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension, selected_files_dir)
            elif method == 'split_all':
                split_all(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension)
            elif method == 'split_all_with_count':
                split_all_with_count(dataset_path, corpus_names, size, dst_dataset_path, extensions, method, output_extension, selected_files_dir)
            else:
                print(f"Unknown method: {method}")
if __name__ == '__main__':
    main()