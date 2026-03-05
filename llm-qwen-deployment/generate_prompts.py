#!/usr/bin/env python3
"""Generate ~1500 synthetic story prompts for load testing.

Combinatorial: 20 characters x 15 moral values x 5 settings = 1500 unique prompts.
Output: prints JSON array to stdout (pipe to file if needed).

Usage:
    python generate_prompts.py > prompts.json
"""

import itertools
import json

CHARACTERS = [
    "rabbit", "bear", "fox", "owl", "turtle",
    "cat", "dog", "lion", "mouse", "elephant",
    "deer", "penguin", "squirrel", "dolphin", "butterfly",
    "bee", "wolf", "frog", "eagle", "lamb",
]

MORALS = [
    "sharing", "honesty", "kindness", "patience", "courage",
    "teamwork", "forgiveness", "gratitude", "perseverance", "respect",
    "empathy", "responsibility", "humility", "generosity", "curiosity",
]

SETTINGS = [
    "forest", "village", "school", "garden", "ocean",
]

TEMPLATE = (
    "Write a single paragraph moral story for children "
    "about a {character} who learns the value of {moral} in a {setting}."
)


def generate_prompts() -> list[str]:
    return [
        TEMPLATE.format(character=c, moral=m, setting=s)
        for c, m, s in itertools.product(CHARACTERS, MORALS, SETTINGS)
    ]


if __name__ == "__main__":
    prompts = generate_prompts()
    print(json.dumps(prompts, indent=2))
