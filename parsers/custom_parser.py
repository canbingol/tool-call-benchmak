from typing import List
import json

def spep_parser(input:str, seperator:str="spep"):
    parts = input.split(seperator)

    tool_jsons = []
    for p in parts:
        p = p.strip()

        if p.startswith("{") and p.endswith("}"):
            tool_jsons.append(p)

    return tool_jsons

def qwen2_5_parser(text: str, separators: List[str] = ["<tool_call>", "</tool_call>"]):

    start, end = separators

    parts = text.split(start)

    tool_jsons = []
    for p in parts:
        p = p.strip()

        if p.endswith(end):
            p = p[:-len(end)]
            tool_jsons.append(json.loads(p))

    return tool_jsons

