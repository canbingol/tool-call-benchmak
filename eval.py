import json

from model import Model
from data import Data

MODEL_ID = "canbingol/qwen2.5-3B_tool_call_1epoch"
DATASET_ID = "Salesforce/xlam-function-calling-60k"


model = Model(model_id=MODEL_ID)

data = Data(dataset_id=DATASET_ID, tokenizer=model.tokenizer, arange=[0, 100], split="train", shuffle=True)
print(data.dataset)

data.apply_chat_template_all()

exact_match = 0.0
for data_item, gold in zip(data.formatted_data, data.golds):

    model_answer = model.generate(data_item["text"])
    
    if model_answer == gold:
        exact_match += 1
    
exact_match /= len(data.formatted_data)
print(exact_match)