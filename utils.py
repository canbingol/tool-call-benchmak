import os
from tqdm import tqdm
from typing import List
import json
from datetime import datetime


def write2file(bench_type:str, golds:List, predictions:List, raw_answer:str, model_name:str, data_size:int):

    folder_name = f"results/{model_name}_{data_size}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = f"{folder_name}/{bench_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    data = []

    try:
        with open(file_name, "r", encoding="UTF-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []
    
    d = {
        "gold": golds,
        "answer": predictions,
        "raw_answer": raw_answer
    }

    data.append(d)

    with open(file_name, "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=3)


def simple_tool_bench(data, model, parser, model_name, limit):
    exact_match = 0.0
    tool_name_accuracy = 0.0
    argument_accuracy = 0.0

    print(f"Running simple tool call benchmark for {model_name} with data size {limit}...")
    for data_item, gold in tqdm(zip(data.formatted_data, data.golds), total=len(data.formatted_data)):

        model_answer = model.generate(data_item["text"])
        tool_call = parser(model_answer)
        
        if tool_call == gold:
            exact_match += 1
            argument_accuracy += 1
            tool_name_accuracy += 1
        else:
            if isinstance(tool_call, list) and len(tool_call) > 0 and tool_call[0]["name"] == gold[0]["name"]:
                tool_name_accuracy += 1

            if isinstance(tool_call, list) and len(tool_call) > 0 and tool_call[0]["arguments"] == gold[0]["arguments"]:
                argument_accuracy += 1

        write2file(bench_type="simple", golds=gold, predictions=tool_call, raw_answer=model_answer,
               model_name=model_name, data_size=limit)

        
    exact_match /= len(data.formatted_data)
    tool_name_accuracy /= len(data.formatted_data)
    argument_accuracy /= len(data.formatted_data)

    return exact_match, tool_name_accuracy, argument_accuracy

def multi_tool_bench(data, model, parser, model_name, limit):
    exact_match = 0.0
    tool_name_accuracy = 0.0
    argument_accuracy = 0.0

    n_tools = 0
    print(f"Running multi tool call benchmark for {model_name} with data size {limit}...")
    for data_item, gold in tqdm(zip(data.formatted_data, data.golds), total=len(data.formatted_data)):

        order_matters = data_item.get("order_matters", True)
        model_answer = model.generate(data_item["text"])
        model_tool_call = parser(model_answer)

        n_tools += len(gold)
        if order_matters:
            if model_tool_call == gold:
                exact_match += 1
                tool_name_accuracy += 1
                argument_accuracy += 1
            
            else:
                if len(model_tool_call) > 0:
                    in_name_acc = 0.0
                    in_args_acc = 0.0

                    for i,tool in enumerate(model_tool_call):
                        if tool["name"] == gold[i]["name"]:
                            in_name_acc += 1
                        if tool["arguments"] == gold[i]["arguments"]:
                            in_args_acc += 1

                    tool_name_accuracy += in_name_acc / len(tool)                  
                    argument_accuracy += in_args_acc / len(tool)                  

        else:

            model_tool_set = set([json.dumps(item, sort_keys=True) for item in model_tool_call])
            gold_set = set([json.dumps(item, sort_keys=True) for item in gold])

            if model_tool_set == gold_set:
                exact_match += 1
                tool_name_accuracy += 1
                argument_accuracy += 1
                      
            else:
                if len(model_tool_call) > 0:
                    in_name_acc = 0.0
                    in_args_acc = 0.0

                    gold_names = [item["name"] for item in gold]
                    gold_args = [item["arguments"] for item in gold]

                    for tool in model_tool_call:
                        name = tool["name"]
                        arguments = tool["arguments"]

                        if name in gold_names:
                            in_name_acc += 1
                        if arguments in gold_args:
                            in_args_acc += 1
                    

                    tool_name_accuracy += in_name_acc / len(tool)
                    argument_accuracy += in_args_acc / len(tool)

        write2file(bench_type="multi", golds=gold, predictions=model_tool_call, raw_answer=model_answer,
               model_name=model_name, data_size=limit)

    exact_match /= len(data.formatted_data)
    tool_name_accuracy /= n_tools
    argument_accuracy /= n_tools

    return exact_match, tool_name_accuracy, argument_accuracy
