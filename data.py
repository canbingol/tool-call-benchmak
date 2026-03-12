from typing import List, Optional
import json
from pprint import pprint

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
class Data:
    def __init__(self, dataset_id: str, tokenizer:PreTrainedTokenizer
                 ,arange:Optional[List[int]], split:str="eval", shuffle:bool=True, seed:int=42):
        
        self.tokenizer = tokenizer
        dataset = load_dataset(dataset_id, split=split)
        if shuffle:
            dataset.shuffle(seed=seed)

        if arange:
            dataset = dataset.select(range(arange[0], arange[1]))


        self.dataset = dataset
        self.formatted_data = None
        self.golds = None

    def __getitem__(self, key):
        return self.dataset[key]
    
    def __len__(self):
        return len(self.dataset)
    
    def apply_chat_template_all(self):
        columns = self.dataset.column_names
        self.formatted_data = self.dataset.map(self._chat_format)
        self.golds = self.formatted_data["answers"]
        self.formatted_data = self.formatted_data.remove_columns(columns)
        
        #self.save2disk(save_path="data/chat_format", dataset=self.formatted_data)
        return 
    def _chat_format(self, example):
        system_message = example.get("system_message", "You are a helpful assistant.")
        user_c = example.get("query")
        assistant_c = json.loads(example.get("answers"))
        tools = json.loads(example.get("tools"))

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_c},
            {"role": "assistant", "content": "", "tool_calls": assistant_c},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False
        )

        return {"text": prompt}
    
    def save2disk(self, save_path:str, dataset:Dataset):
        try:
            dataset.to_parquet(save_path)
            return 0
        except:
            return 1
        
