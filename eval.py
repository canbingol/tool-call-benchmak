import json
from argparse import ArgumentParser

from models.hf_model import Model
from data import Data
from parsers.custom_parser import spep_parser, qwen2_5_parser


parser = ArgumentParser()

parser.add_argument("--model_id", type=str)
parser.add_argument("--data_size", type=int)
parser.add_argument(
    "--parser",
    type=str,
    choices=["spep", "qwen2_5"],
)

args = parser.parse_args()

MODEL_ID = args.model_id
limit = args.data_size
pars = args.parser

PARSER = spep_parser if pars == "spep" else qwen2_5_parser
DATASET_ID = "Salesforce/xlam-function-calling-60k"


model = Model(model_id=MODEL_ID)

data = Data(dataset_id=DATASET_ID, tokenizer=model.tokenizer, arange=[0, limit], split="train", shuffle=True)
print(data.dataset)

data.apply_chat_template_all()

exact_match = 0.0
for data_item, gold in zip(data.formatted_data, data.golds):

    model_answer = model.generate(data_item["text"])
    tool_call = PARSER(model_answer)

    if tool_call == gold:
        exact_match += 1
    
    
exact_match /= len(data.formatted_data)
print(exact_match)