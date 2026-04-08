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

        if end in p:
            p = p.split(end, 1)[0].strip()
            try:
                tool_jsons.append(json.loads(p))
            except json.JSONDecodeError:
                tool_jsons.append(None)
                
    return tool_jsons

