import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map keys in selector json (e.g. 1.json) to values from block json files. "
            "Example: selector[key]=18 -> output[key]=18.json[key]."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing selector and block json files, e.g. Predictor/utils/gsm8k_0T0.0",
    )
    parser.add_argument(
        "--selector-file",
        type=str,
        default="1.json",
        help="Selector filename inside input_dir (default: 1.json)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mapped_from_selector.json",
        help="Output filename inside input_dir (default: mapped_from_selector.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    selector_path = os.path.join(input_dir, args.selector_file)
    output_path = os.path.join(input_dir, args.output_file)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not os.path.isfile(selector_path):
        raise FileNotFoundError(f"Selector file not found: {selector_path}")

    with open(selector_path, "r", encoding="utf-8") as f:
        selector = json.load(f)

    block_cache: dict[int, dict] = {}
    result: dict[str, float] = {}

    for key, block_value in selector.items():
        try:
            block_size = int(block_value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid block size for key {key}: {block_value}")

        if block_size not in block_cache:
            block_path = os.path.join(input_dir, f"{block_size}.json")
            if not os.path.isfile(block_path):
                raise FileNotFoundError(f"Block file not found: {block_path}")
            with open(block_path, "r", encoding="utf-8") as f:
                block_cache[block_size] = json.load(f)

        block_data = block_cache[block_size]
        if key not in block_data:
            raise KeyError(f"Key {key} not found in {block_size}.json")

        result[key] = block_data[key]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Wrote mapped json: {output_path}")


if __name__ == "__main__":
    main()
