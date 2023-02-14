import argparse
import glob
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="raw-data",
                        help="Path to raw data folder")
    parser.add_argument("--output", type=str, default="outputs/combined-raw-data.json",
                        help="File to output")

    return parser.parse_args()


def main(args):

    result = {}
    for test in os.listdir(args.raw_data_dir):
        test_path = os.path.join(args.raw_data_dir, test)
        if not os.path.isdir(test_path):
            continue

        result[test] = {
            'sections': []
        }

        section_paths = glob.glob(os.path.join(test_path, "section_*"))
        for section_num in range(1, len(section_paths) + 1):
            section_path = os.path.join(test_path, f"section_{section_num}")

            section = {
                'questions': {},
                'passages': [open(os.path.join(
                    section_path, f"passage_{i}.txt"), "r", encoding="utf-8").read() for i in range(1, len(glob.glob(
                        os.path.join(section_path, "passage_*.txt"))) + 1)],
                'tables': [open(os.path.join(
                    section_path, f"table_{i}.txt"), "r", encoding="utf-8").read() for i in range(1, len(glob.glob(
                        os.path.join(section_path, "table_*.txt"))) + 1)],
                'graphs': [open(os.path.join(
                    section_path, f"graph_{i}.txt"), "r", encoding="utf-8").read() for i in range(1, len(glob.glob(
                        os.path.join(section_path, "graph_*.txt"))) + 1)],
                'figures': [open(os.path.join(
                    section_path, f"figure_{i}.txt"), "r", encoding="utf-8").read() for i in range(1, len(glob.glob(
                        os.path.join(section_path, "figure_*.txt"))) + 1)],
                'context': open(os.path.join(section_path, "context.txt"), "r", encoding="utf-8").read()
            }

            for question_path in glob.glob(os.path.join(section_path, "question_*.txt")):
                full_key = question_path[question_path.rindex("_")+1:-4]
                dash = full_key.find("-")

                lines = open(question_path, "r", encoding="utf-8").readlines()

                section['questions'][full_key if dash == -1 else full_key[0:dash]] = {
                    'text': lines[0].strip(),
                    'answers': [x.replace("*", "").strip() for x in lines[1:]],
                    'answer': lines[[x[0] for x in lines[1:]].index('*')+1][1],
                    'requires': full_key[dash+1:].split('-') if dash != -1 else [],
                }

            result[test]['sections'].append(section)

    json.dump(result, open(args.output, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
