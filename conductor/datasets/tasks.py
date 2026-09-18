"""Disjoint deterministic task families for inexpensive end-to-end verification."""
from __future__ import annotations
import hashlib
import json
import random
from conductor.schema import Task

TEMPLATE_FAMILIES = ("arithmetic_addition", "inline_lookup", "reverse_string", "count_vowels",
                     "lookup_multiply", "addition_reverse_digits")


def _id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def make_tasks(seed: int = 42, train_count: int = 24, eval_count: int = 12) -> list[Task]:
    if train_count < 1 or eval_count < 1:
        raise ValueError("both splits must contain tasks")
    randomizer = random.Random(seed)
    tasks: list[Task] = []
    words = ["orbit", "signal", "cobalt", "river", "meadow", "silver", "violet", "amber"]
    global_index = 0
    for split, count in (("train", train_count), ("eval", eval_count)):
        for index in range(count):
            # Distinct public operands/text give genuine prompt separation, without split labels.
            global_index += 1
            a, b = global_index * 101 + randomizer.randint(2, 99), randomizer.randint(2, 17)
            category = index % (4 if split == "train" else 6)
            if category == 0:
                task_type, text, answer = "math", f"Compute {a} + {b}.", str(a + b)
            elif category == 1:
                facts = json.dumps({"Orion": a, "Lyra": b, "Vega": a + b}, sort_keys=True)
                task_type, text, answer = "retrieval", f"What is Orion's value? Reference: {facts}", str(a)
            elif category in {2, 3}:
                number = global_index
                identifier = ""
                while number:
                    number, digit = divmod(number - 1, 26)
                    identifier = chr(97 + digit) + identifier
                word = words[index % len(words)] + " " + words[randomizer.randrange(len(words))] + " " + identifier
                task_type = "code"
                if category == 2:
                    text, answer = f'Reverse the text "{word}".', word[::-1]
                else:
                    text, answer = f'Count vowels in the text "{word}".', str(sum(ch in "aeiou" for ch in word))
            elif category == 4:
                facts = json.dumps({"Orion": a, "Lyra": b}, sort_keys=True)
                task_type, text, answer = "composed_retrieval_math", f"Find Orion's value in reference, then multiply that value by {b}. Reference: {facts}", str(a * b)
            else:
                task_type, text, answer = "composed_math_string", f"Compute {a} + {b}. then reverse the digits of the result.", str(a + b)[::-1]
            tasks.append(Task(_id(text), text, task_type, split, answer,
                              {"family": task_type, "template_family": TEMPLATE_FAMILIES[category],
                               "ood": category >= 4, "grader": "exact", "generation_seed": seed}))
    if len({_id(task.user_task) for task in tasks}) != len(tasks):
        raise RuntimeError("duplicate task text")
    return tasks
