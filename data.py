from typing import List, Optional, Sequence
import json

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class Data:
    def __init__(
        self,
        dataset_id: str,
        tokenizer: PreTrainedTokenizer,
        arange: Optional[Sequence[int]] = None,
        split: str = "eval",
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer

        if dataset_id.endswith(".json"):
            dataset = Dataset.from_json(dataset_id)
        else:
            dataset = load_dataset(dataset_id, split=split)

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        if arange is not None and len(arange) == 2:
            start, end = arange
            if end == -1:
                dataset = dataset.select(range(start, len(dataset)))
            else:
                dataset = dataset.select(range(start, end))

        self.dataset = dataset
        self.formatted_data = None
        self.golds = None

    def __getitem__(self, key):
        return self.dataset[key]

    def __len__(self):
        return len(self.dataset)

    def _parse_tools(self, tools_raw):
        if tools_raw is None:
            return []

        if isinstance(tools_raw, str):
            return json.loads(tools_raw)

        if isinstance(tools_raw, list):
            return tools_raw

        raise TypeError(f"Unsupported tools type: {type(tools_raw)}")

    def _parse_answers(self, answers_raw):
        if answers_raw is None:
            return None

        if isinstance(answers_raw, str):
            return json.loads(answers_raw)

        return answers_raw

    def apply_chat_template_all(self):
        original_columns = self.dataset.column_names

        self.golds = [
            self._parse_answers(example.get("answers"))
            for example in self.dataset
        ]

        self.formatted_data = self.dataset.map(self._chat_format)
        self.formatted_data = self.formatted_data.remove_columns(original_columns)

    def _chat_format(self, example):
        system_message = example.get("system_message", "You are a helpful assistant.")
        user_content = example.get("query", "")
        tools = self._parse_tools(example.get("tools"))

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {"text": prompt}
        
