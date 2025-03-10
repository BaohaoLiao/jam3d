import os
import json
from pathlib import Path
from typing import Iterable, Union, Any


PROMPT_TEMPLATES = {
    "deepseek-r1": (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}."
        "<｜User｜>{input}<｜Assistant｜><think>\n",
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-no-think": (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}."
        "<｜User｜>{input}<｜Assistant｜><think>\n\n</think>",
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-choice": (
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{input}<｜Assistant｜><think>\n",
        "{output}",
        "\n\n",
    ),
     "deepseek-r1-no-think-choice": (
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{input}<｜Assistant｜><think>\n\n</think>",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
}


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    assert os.path.exists(data_file)
    examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def construct_prompt(example, data_name, args):
    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]
    input_template, output_template, splitter = prompt_temp[0], prompt_temp[1], prompt_temp[2]
    full_prompt = input_template.format(input=example["question"])
    return full_prompt.strip(" ")