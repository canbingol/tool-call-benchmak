from typing import List
import json
from datetime import datetime


def write2file(golds:List, predictions:List, raw_answer:str, model_name:str, data_size:int):
    file_name = f"{model_name}_{data_size}_{datetime.now().strftime('%Y%m%d')}"
    data = []

    try:
        with open(file_name, "r", encoding="UTF-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []
    
    d = {
        "gold": json.loads(golds),
        "answer": json.loads(predictions),
        "raw_answer": raw_answer
    }

    data.append(d)

    with open(file_name, "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=3)


