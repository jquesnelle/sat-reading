import argparse
import json
import openai
import os
import sys
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model")
    parser.add_argument("model_type", type=str, help="Type of model", choices=[
                        'seq2seq', 'causallm', 'openai'])
    parser.add_argument("--dtype", type=str, help="Data type to load model in",
                        choices=['float32', 'float16', 'bfloat16', '8bit'], default="bfloat16")

    return parser.parse_args()


def model_args(args):
    if args.dtype == "8bit":
        return {"load_in_8bit": True}
    elif args.dtype == "float32":
        return {"torch_dtype": torch.float32}
    elif args.dtype == "float16":
        return {"torch_dtype": torch.float16}
    elif args.dtype == "bfloat16":
        return {"torch_dtype": torch.bfloat16}
    else:
        raise RuntimeError(f"Unknown dtype {args.dtype}")


def run_transformers_model(tokenizer, model, generate_args, does_echo):
    while True:
        input_text = input()
        input_ids = tokenizer(
            input_text, return_tensors="pt").input_ids.to("cuda")

        if len(input_ids) >= 2048:
            print("Warning: tokenized prompt >= 2048 tokens")

        outputs = model.generate(input_ids, **generate_args)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if does_echo:
            answer = answer[input_text:]

        print(answer)


def seq2seq(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, device_map="auto", **model_args(args))

    return run_transformers_model(tokenizer, model, {"max_length": 512}, False)


def causallm(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", **model_args(args))

    return run_transformers_model(tokenizer, model, {"max_new_tokens": 512}, True)


def main(args):
    if args.model_type == "seq2seq":
        seq2seq(args)
    elif args.model_type == "causallm":
        causallm(args)
    else:
        raise RuntimeError(f"Unknown model type {args.model_type}")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
