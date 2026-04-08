import os
from tqdm import tqdm
from typing import List
import json
from datetime import datetime


def write2file(
    bench_type: str,
    golds: List,
    predictions: List,
    raw_answer: str,
    model_name: str,
    data_size: int,
    run_id: str | None = None,
):
    safe_model_name = model_name.replace("/", "__").replace("\\", "__").replace(":", "_")

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = os.path.join("results", f"{safe_model_name}_{data_size}")
    os.makedirs(folder_name, exist_ok=True)

    file_name = os.path.join(folder_name, f"{bench_type}_{run_id}.jsonl")

    record = {
        "gold": golds,
        "answer": predictions,
        "raw_answer": raw_answer,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(file_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def simple_tool_bench(data, model, parser, model_name, limit, run_id):
    exact_match = 0.0

    print(f"Running simple tool call benchmark for {model_name} with data size {limit}...")
    for data_item, gold in tqdm(zip(data.formatted_data, data.golds), total=len(data.formatted_data)):

        model_answer = model.generate(data_item["text"])
        tool_call = parser(model_answer)
        
        if tool_call == gold:
            exact_match += 1
            tool_accuracy += 1

        write2file(bench_type="simple", golds=gold, predictions=tool_call, raw_answer=model_answer,
               model_name=model_name, data_size=limit, run_id=run_id)

    exact_match /= len(data.formatted_data)

    return exact_match

def multi_tool_bench(data, model, parser, model_name, limit, run_id=None):
    exact_match = 0.0
    tool_accuracy = 0.0

    n_tools = 0
    print(f"Running multi tool call benchmark for {model_name} with data size {limit}...")
    for data_item, gold in tqdm(zip(data.formatted_data, data.golds), total=len(data.formatted_data)):

        order_matters = data_item.get("order_matters", True)
        model_answer = model.generate(data_item["text"])
        model_tool_call = parser(model_answer)
        
        if not isinstance(model_tool_call, list):
            model_tool_call = []

        n_tools += len(gold)
        if order_matters:
            if model_tool_call == gold:
                exact_match += 1
                tool_accuracy += len(gold)
            
            else:
                if len(model_tool_call) > 0:

                    tool_len = min(len(model_tool_call), len(gold))
                    in_tool_acc = 0.0

                    for i,tool in enumerate(model_tool_call):
                        if i >= tool_len:
                            break

                        tool_name = tool.get("name", "")
                        tool_args = tool.get("arguments", {})   

                        if tool_name == gold[i].get("name") and tool_args == gold[i].get("arguments"):
                                in_tool_acc += 1

                    tool_accuracy += in_tool_acc                           

        else:

            model_tool_set = sorted([json.dumps(item, sort_keys=True) for item in model_tool_call])
            gold_set = sorted([json.dumps(item, sort_keys=True) for item in gold])

            if model_tool_set == gold_set:
                exact_match += 1
                tool_accuracy += len(gold)
                      
            else:
                if len(model_tool_call) > 0:
                    in_tool_acc = 0.0

                    remaining_gold = gold.copy()

                    for tool in model_tool_call:
                        matched_idx = None
                        for i, gold_tool in enumerate(remaining_gold):
                            if tool == gold_tool:
                                matched_idx = i
                                break

                        if matched_idx is not None:
                            in_tool_acc += 1
                            remaining_gold.pop(matched_idx)

                    tool_accuracy += in_tool_acc
                    

        write2file(bench_type="multi", golds=gold, predictions=model_tool_call, raw_answer=model_answer,
               model_name=model_name, data_size=limit, run_id=run_id)

    exact_match /= len(data.formatted_data)
    tool_accuracy /= n_tools

    return exact_match, tool_accuracy
