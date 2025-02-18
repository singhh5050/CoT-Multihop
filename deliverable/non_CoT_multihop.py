"""
Performing an run of the Inspect AI framework using 5 question-answer pairs taken from the MoreHopQA: More Than Multi-hop Reasoning paper. This will test models on their ability to shift between arithmetic, symbolic, and commonsense reasoning.

The initial dataset is quite large, so in efforts to reduce API calls, I downsampled to 5 types of questions:
> Q1: Symbolic, Commonsense
> Q2: Commonsense, Arithmetic
> Q3: Symbolic, Arithmetic
> Q4: Purely Symbolic
> Q5: Purely Arithmetic

This run DOES NOT include Chain of Thought reasoning. I have left out that solving process so that generate() can serve as a baseline.
I also edited the system prompt to make sure that it just spits out the answer without any reasoning.
"""

# don't forget: 
# pip install inspect-ai python-dotenv

# The following packages don't run on Python 3.12 -- I had to switch my interpreter to 3.11.7
from inspect_ai import Task, eval, task
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message
from inspect_ai.dataset import json_dataset


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

SYSTEM_MESSAGE = """
Please just give me the answer to the question. Don't include any reasoning. Just spit out the answer as quickly as possible.
"""

# I personally like to use VS Code to run task directly from .py file, rather than inputting inspect eval command in terminal -- it's more reliable
@task
def non_CoT_multihop():
    return Task(
        dataset=json_dataset("/Users/harshsingh/Desktop/Multihop Reasoning + CoT/deliverable/moreHopQA_downsampled.json"),
        solver=[system_message(SYSTEM_MESSAGE), generate()], # no CoT, but with system message
        scorer=model_graded_fact(),
    )