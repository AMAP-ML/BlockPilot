import os
import json

root_dir = "utils"


def is_valid_subfolder(subfolder_path):
    files = os.listdir(subfolder_path)

    # ❌  json  → 
    if any(not f.endswith(".json") for f in files):
        return False

    return True


def process_subfolder(subfolder_path):
    files = [
        f for f in os.listdir(subfolder_path)
        if f.endswith(".json") and f not in ("0.json", "1.json")
    ]

    if not files:
        return None

    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    data = {}

    for file in files:
        file_path = os.path.join(subfolder_path, file)
        with open(file_path, "r") as f:
            data[file] = json.load(f)

    # （0）
    win_count = {file: 0 for file in data.keys()}

    keys = list(next(iter(data.values())).keys())

    for k in keys:
        max_file = None
        max_value = float("-inf")

        for file, content in data.items():
            value = content[k]
            if value > max_value:
                max_value = value
                max_file = file

        win_count[max_file] += 1

    return win_count, len(keys)


# 
subfolders = [
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f))
]

for subfolder in sorted(subfolders):
    subfolder_path = os.path.join(root_dir, subfolder)

    if not is_valid_subfolder(subfolder_path):
        print(f"⛔ : {subfolder}")
        continue

    result = process_subfolder(subfolder_path)

    if result is None:
        print(f"⚠️ (): {subfolder}")
        continue

    win_count, total_keys = result

    print(f"\n📂 : {subfolder}")

    for k in sorted(win_count.keys(), key=lambda x: int(x.split(".")[0])):
        print(f"{k}: {win_count[k]}")

    # ✅  16.json 
    if "16.json" in win_count:
        ratio = win_count["16.json"] / total_keys
        print(f"👉 16.json : {ratio:.2f} ({win_count['16.json']}/{total_keys})")
    else:
        print("⚠️ 16.json ")