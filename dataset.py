import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import DraftModel, extract_context_feature, sample


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def load_txt_dataset(file_path: str) -> list[dict]:
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append({"turns": [line]})
    return dataset


@torch.inference_mode()
def draft_generate(
    model: DraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    collect_hidden_states: bool = False,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    prefill_logits = output.logits.detach().cpu() if collect_hidden_states else None

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        prefill_logits=prefill_logits,
    )


def format_temp(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to txt file, one prompt per line.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Output dataset name. Defaults to txt filename stem.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root dir, e.g. utils")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 14, 15, 16, 17, 18])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save hidden states to output_dir/_hidden_states.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_txt_dataset(args.input_file)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    if not dataset:
        raise ValueError(f"No valid non-empty lines found in {args.input_file}")

    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.input_file))[0]
    temp_str = format_temp(args.temperature)
    run_dir = os.path.join(args.output_dir, f"{dataset_name}_{args.seed}T{temp_str}")
    os.makedirs(run_dir, exist_ok=True)
    if args.save_hidden_states:
        os.makedirs(os.path.join(args.output_dir, "_hidden_states"), exist_ok=True)

    for block_size in args.block_sizes:
        print(f"Running block_size={block_size}")
        acceptance_dict: dict[int, float] = {}
        hidden_states_dict: dict[int, torch.Tensor] = {}

        for idx, instance in enumerate(tqdm(dataset)):
            messages = []
            for user_content in instance["turns"]:
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

                response = draft_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    collect_hidden_states=args.save_hidden_states,
                )

                acceptance_dict[idx] = sum(response.acceptance_lengths) / len(response.acceptance_lengths)
                if args.save_hidden_states and response.prefill_logits is not None:
                    hidden_states_dict[idx] = response.prefill_logits

                generated_ids = response.output_ids[0, response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})

        json_path = os.path.join(run_dir, f"{block_size}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(acceptance_dict, f)

        if args.save_hidden_states:
            hidden_path = os.path.join(
                args.output_dir,
                "_hidden_states",
                f"{dataset_name}_seed{args.seed}T{temp_str}samples{len(dataset)}_bs{block_size}.pt",
            )
            torch.save(hidden_states_dict, hidden_path)

    print(f"Done. Acceptance JSON saved to: {run_dir}")
    if args.save_hidden_states:
        print(f"Hidden states saved to: {os.path.join(args.output_dir, '_hidden_states')}")


if __name__ == "__main__":
    main()
