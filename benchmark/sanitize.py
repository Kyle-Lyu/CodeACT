"""Purpose of this file: Sanitize the code produced by LLMs for the following reasons.
1. Vicuna generated code could miss one white space. We fix the white space to make Vicuna more capable.
2. {Our fault lol.} We find more EOFs tokens afterwards and truncate some messy code afterwards.
"""
import argparse
import ast
import re
import traceback
from typing import List, Optional

from evalplus.data import (
    get_human_eval_plus,
    get_mbpp_plus,
    load_solutions,
    write_jsonl,
)


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False
    

def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def remove_unindented_lines(code, protect_before, execeptions, trim_tails):
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in execeptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            # cut off everything behind
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code


def sanitize(
    old_code: str,
    entry_point: str,
    rm_prefix_lines: Optional[str] = None,
    eofs: List = None,
):
    new_code = old_code
    if rm_prefix_lines is not None:
        new_code = "\n".join(
            [
                line
                for line in old_code.splitlines()
                if not line.startswith(rm_prefix_lines)
            ]
        )

    new_code = "\n" + new_code
    def_left = "def " + entry_point

    # basic handling of chat output
    new_code = new_code.replace("\n```python\n", "\n```\n")
    for chunk in new_code.split("\n```"):
        if def_left in chunk:
            new_code = chunk
            break

    chunks = [chunk for chunk in re.split(f"{def_left}\s*\(", new_code)]
    # TODO: having return does not mean this is complete
    bodies = [chunk for chunk in chunks[1:] if "  return " in chunk.split("\ndef")[0]]
    def_left = def_left + "("
    new_code = def_left + def_left.join(bodies) if len(bodies) > 0 else ""  # fn + impl
    new_code = to_four_space_indents(new_code)

    for eof in eofs or []:
        new_code = new_code.split(eof)[0]

    # remove lines starting from the first unindented line after def_left
    new_code = remove_unindented_lines(
        new_code,
        protect_before=def_left,
        execeptions=["def ", "import ", "from "],
        trim_tails=['"""', "if", "print"],
    )
    new_code = chunks[0] + new_code

    # cut all functions that are not syntactically correct && not the entry point
    parts = new_code.split("\ndef ")
    includes = [parts[0]]
    for fn in new_code.split("\ndef ")[1:]:
        if (
            fn.strip().startswith(entry_point + " ")
            or fn.strip().startswith(entry_point + "(")
            or syntax_check("\ndef " + fn)
        ):
            includes.append(fn)
    new_code = "\ndef ".join(includes)
    return new_code.strip()


EOS = [
    "\nif __name__", "\ndef main(",
    "\nprint(", "\nassert",
    "\ndef ", "\nclass ", "\nimport ", "\nfrom ", '\n"""'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_file", type=str, required=True,
    )
    parser.add_argument(
        "--debug-task", type=str, 
        default=None, help="Enter the task ID to only sanitize that task."
    )
    args = parser.parse_args()

    # task_id -> entry_point
    entry_point = {}
    # merge two datasets
    dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
    
    for task_id, problem in dataset.items():
        entry_point[task_id] = problem["entry_point"]

    assert args.sample_file.endswith(".jsonl")
    output_file = args.sample_file.replace(".jsonl", "-sanitized.jsonl")

    nsan = 0
    ntotal = 0
    new_solutions = []
    for solution in load_solutions(args.sample_file):
        task_id = solution["task_id"]
        dbg_identifier = solution["_identifier"]

        if args.debug_task is not None and task_id != args.debug_task:
            continue

        ntotal += 1
        old_code = solution["solution"]
        old_code = old_code.strip()

        new_code = sanitize(
            old_code=old_code,
            entry_point=entry_point[task_id],
            eofs=EOS,
        ).strip()
        new_code = code_extract(new_code).strip()

        # if changed, print the message
        if new_code != old_code:
            msg = "Sanitized: " + dbg_identifier
            print(msg)
            nsan += 1

        new_solutions.append({"task_id": task_id, "solution": new_code})

    write_jsonl(output_file, new_solutions)

    print(f"Sanitized {nsan} out of {ntotal} files.")
    print(f"Sanitized solutions at {output_file}")


if __name__ == "__main__":
    main()
