from argparse import ArgumentParser
from datetime import datetime
from models.hf_model import Model
from data import Data
from parsers.custom_parser import spep_parser, qwen2_5_parser
from utils import simple_tool_bench, multi_tool_bench

parser = ArgumentParser()

parser.add_argument("--model_id", type=str)
parser.add_argument("--data_size", type=int, default=-1)
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
SIMPLE_DATASET_ID = "./eval_data/simple_tool_call.json"
MULTI_DATASET_ID = "./eval_data/multi_tool_call.json"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


model = Model(model_id=MODEL_ID)

simple_data = Data(dataset_id=SIMPLE_DATASET_ID, tokenizer=model.tokenizer, arange=[0, limit], split="train", shuffle=True)
simple_data.apply_chat_template_all()

multi_data = Data(dataset_id=MULTI_DATASET_ID, tokenizer=model.tokenizer, arange=[0, limit], split="train", shuffle=True)
multi_data.apply_chat_template_all()

model_name = MODEL_ID.split("/")[-1]

simple_exact_match, tool_name_accuracy, argument_accuracy = simple_tool_bench(simple_data, model, PARSER, model_name, limit, run_id)

print(f"| TOOL CALL BENCHMARK | {model_name} | data size: {limit} |")
print("---" * 30)
print(f"| SİMPLE TOOL CALL | Exact Match | Tool Name Accuracy | Argument Accuracy |")
print("---" * 30)
print(f"| Exact Match: {simple_exact_match:.4f} | Tool Name Accuracy: {tool_name_accuracy:.4f} | Argument Accuracy: {argument_accuracy:.4f} |")
print("---" * 30)

multi_exact_match, multi_tool_name_accuracy, multi_argument_accuracy = multi_tool_bench(multi_data, model, PARSER, model_name, limit, run_id)
print(f"| MULTI TOOL CALL | Exact Match | Tool Name Accuracy | Argument Accuracy |")
print("---" * 30)
print(f"| Exact Match: {multi_exact_match:.4f} | Tool Name Accuracy: {multi_tool_name_accuracy:.4f} | Argument Accuracy: {multi_argument_accuracy:.4f} |")