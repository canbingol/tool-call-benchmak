from typing import List

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

    parts = input.split(start)

    tool_jsons = []
    for p in parts:
        p = p.strip()

        if p.endswith(end):
            tool_jsons.append(p)

    return tool_jsons

