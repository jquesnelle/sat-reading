import argparse
import json
import os
import sys
from typing import Any, Dict, List


def create_dataset_for_keys(combined_raw_data: Dict, args: Any, keys: List[str], output_file: str):
    data = []
    for key in keys:
        for section in combined_raw_data[key]["sections"]:

            header = f"SAT READING COMPREHENSION TEST\n\n{section['context']}\n\n"
            offset = 1
            for i in range(0, len(section["passages"])):
                if i > 0:
                    header += "\n\n"
                if len(section["passages"]) > 1:
                    header += f"Passage {i + 1}\n"

                passage = section["passages"][i]
                if args.line_numbers == 0:
                    passage = "\n".join(
                        [f"{str(index + offset).rjust(3, ' ')}{line}" for index, line in enumerate(passage.split('\n'))])
                elif args.line_numbers == 1:
                    passage = "\n".join(
                        [f"{str(index + offset).rjust(3, ' ') if ((index + offset) % 5) == 0 else '   '}{line}" for index, line in enumerate(passage.split('\n'))])

                header += passage

                offset += len(passage.split('\n'))

            for i in range(0, len(section["tables"])):
                header += "\n\n"
                if len(section["tables"]) > 1:
                    header += f"Table {i + 1}\n"
                header += f"{section['tables'][i]}"

            for i in range(0, len(section["graphs"])):
                header += "\n\n"
                if len(section["graphs"]) > 1:
                    header += f"Graph {i + 1}\n"
                header += f"{section['graphs'][i]}"

            for i in range(0, len(section['figures'])):
                header += "\n\n"
                if len(section["figures"]) > 1:
                    header += f"Figure {i + 1}\n"
                header += f"{section['figures'][i]}"

            for num, question in section["questions"].items():
                if not args.include_previous and "previous" in question["requires"]:
                    continue
                if not args.include_graph and "graph" in question["requires"]:
                    continue
                if not args.include_table and "table" in question["requires"]:
                    continue
                if not args.include_figure and "figure" in question["requires"]:
                    continue
                if not args.include_line and "line" in question["requires"]:
                    continue
                answers = "\n".join(question["answers"])
                data.append({
                    "text": f"{header}\n\n\n\nQuestion {num}:\n{question['text']}\n{answers}\n\nAnswer:",
                    "answer": question["answer"],
                    "requires_line" if args.requires_line_feature else "requires": "line" in question["requires"] if args.requires_line_feature else question["requires"],
                    "id": f"{key}-question_{num}"
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    open(output_file, "w", encoding="utf-8").writelines(
        "\n".join([json.dumps(line) for line in data]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to output to")
    parser.add_argument("--combined_raw_data", type=str, default="outputs/combined-raw-data.json",
                        help="Path to combined raw data")
    parser.add_argument("--test", type=str, default=None,
                        help="Key to use for test dataset")
    parser.add_argument("--validation", type=str, default=None,
                        help="Key to use for validation dataset")
    parser.add_argument("--line_numbers", type=int, default=2,
                        help="Line numbers to generate: 0 for full, 1 for SAT-style (every 5), 2 for none")
    parser.add_argument("--include_previous", type=bool, default=False,
                        help="Include questions that depend on the previous question")
    parser.add_argument("--include_graph", type=bool, default=False,
                        help="Include questions that depend on a graph")
    parser.add_argument("--include_table", type=bool, default=False,
                        help="Include questions that depend on a table")
    parser.add_argument("--include_figure", type=bool, default=False,
                        help="Include questions that depend on a figure")
    parser.add_argument("--include_line", type=bool, default=True,
                        help="Include questions that depend on a line number")
    parser.add_argument("--requires_line_feature", type=bool, default=False,
                        help="Only use a single bool requires_line feature")

    return parser.parse_args()


def main(args):
    combined_raw_data: Dict[Any] = json.load(
        open(args.combined_raw_data, "r", encoding="utf-8"))

    keys = [key for key in combined_raw_data.keys()]

    if args.test is not None:
        keys.remove(args.test)
        create_dataset_for_keys(combined_raw_data, args, [args.test], os.path.join(
            args.output_dir, args.dataset_name, "data", "test.jsonl"))

    if args.validation is not None:
        keys.remove(args.validation)
        create_dataset_for_keys(combined_raw_data, args, [args.validation], os.path.join(
            args.output_dir, args.dataset_name, "data", "valid.jsonl"))

    create_dataset_for_keys(combined_raw_data, args, keys, os.path.join(
        args.output_dir, args.dataset_name, "data", "train.jsonl"))


if __name__ == "__main__":
    sys.exit(main(parse_args()))
