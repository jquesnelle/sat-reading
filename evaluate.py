import argparse
import json
import openai
import os
import sys
import torch
import time
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model")
    parser.add_argument("model_type", type=str, help="Type of model", choices=[
                        'seq2seq', 'causallm', 'openai'])
    parser.add_argument("dataset", type=str, help="Dataset")
    parser.add_argument("--dtype", type=str, help="Data type to load model in",
                        choices=['float32', 'float16', 'bfloat16', '8bit'], default="bfloat16")
    parser.add_argument("--split", type=str, help="Split to run on",
                        choices=['train', 'test', 'valid'], default='train')
    parser.add_argument("--result_file", type=str,
                        help="Path to file to record results in", default="outputs/results.json")

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


def new_results():
    def results():
        return {
            "num_questions": 0,
            "correct": 0
        }

    return {
        "all": results(),
        "line": results(),
        "graph": results(),
        "table": results(),
        "figure": results()
    }


def score(results, question, answer):
    results['num_questions'] += 1
    results['correct'] += 1 if question['answer'].upper() == answer.upper() else 0


def score_answer(results, question, answer):
    answer = answer.strip()
    if answer.startswith("Choice "):  # ada seems to do this sometimes
        answer = answer[7:]

    final_answer = answer[0] if len(answer) > 0 else ' '
    if final_answer not in ['A', 'B', 'C', 'D']:
        print(f"Unexpected answer format: {answer}")

    score(results["all"], question, final_answer)
    for requires in question["requires"]:
        score(results[requires], question, final_answer)


def run_transformers_model(dataset, tokenizer, model, generate_args, does_echo):
    results = new_results()

    for question in tqdm(dataset):
        input_ids = tokenizer(
            question["text"], return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids, **generate_args)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if does_echo:
            answer = answer[len(question["text"]):]

        score_answer(results, question, answer)

    return results


def seq2seq(args):
    dataset = load_dataset(args.dataset)[args.split]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, device_map="auto", **model_args(args))

    return run_transformers_model(dataset, tokenizer, model, {"max_length": 32}, False)


def causallm(args):
    dataset = load_dataset(args.dataset)[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", **model_args(args))

    return run_transformers_model(dataset, tokenizer, model, {"max_new_tokens": 32}, True)


def oai(args):
    dataset = load_dataset(args.dataset)[args.split]
    results = new_results()

    for question in tqdm(dataset):
        completion = openai.Completion.create(
            engine=args.model, prompt=question["text"], temperature=0)
        answer = completion['choices'][0]['text']

        score_answer(results, question, answer)

        time.sleep(1)  # you know... the rate limit...

    return results


def main(args):
    if args.model_type == "seq2seq":
        result = seq2seq(args)
    elif args.model_type == "causallm":
        result = causallm(args)
    elif args.model_type == "openai":
        result = oai(args)
    else:
        raise RuntimeError(f"Unknown model type {args.model_type}")

    scores = {}
    for key, value in result.items():
        if value['num_questions'] > 0:
            percent_correct = round(
                (value['correct'] / value['num_questions'] * 100), 1)
            scores[key] = percent_correct

    if os.path.exists(args.result_file):
        results = json.load(open(args.result_file, "r", encoding="utf-8"))
    else:
        results = []

    results.append({
        "model": args.model,
        "dataset": args.dataset,
        "dtype": args.dtype,
        "split": args.split,
        "time": int(time.time()),
        "scores": scores
    })

    json.dump(results, open(args.result_file, "w", encoding="utf-8"), indent=2)

    print(scores)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
