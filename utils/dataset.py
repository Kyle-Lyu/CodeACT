import json
import copy
import torch
from torch.utils.data import Dataset
import transformers
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union


IGNORE_TOKEN_ID = -100
PROMPT_TEMPLATE = {
    "codellama": "[INST] {text} [/INST]",
    "codegemma": "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n",
    "deepseek": "### Instruction:\n{text}\n### Response:\n",
    "starcoder2": "### Instruction:\n{text}\n### Response:\n",
    "codeqwen": "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
    "llama3": "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
}


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


@dataclass
class DataCollatorWithDynamicPad:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable value is "pt".
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_TOKEN_ID
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.return_tensors == "pt", 'Only support Torch framework, set return_tensors="pt"'

        # The inputs' length can't exceed `max_length`
        for feature in features:
            for key in feature.keys():
                feature[key] = feature[key][:self.max_length]
        
        labels = [feature["labels"] for feature in features]
        no_labels_features = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        
        if padding_side == "right":
            batch["labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        return batch
    

@dataclass
class DataCollatorWithDynamicPack:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable value is "pt".
    """

    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = IGNORE_TOKEN_ID
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.return_tensors == "pt", 'Only support Torch framework, set return_tensors="pt"'

        # The inputs' length can't exceed `max_length`
        for feature in features:
            for key in feature.keys():
                feature[key] = feature[key][:self.max_length]

        # sort features by input_ids length
        features.sort(key=lambda x: len(x["input_ids"]))

        batched_features = list()
        packed_sample = {
            "input_ids": list(),
            "attention_mask": list(),
            "labels": list(),
        }
        for feature in features:
            if (len(packed_sample["input_ids"]) + len(feature["input_ids"])) > self.max_length:
                batched_features.append(copy.deepcopy(packed_sample))
                packed_sample.clear()
                packed_sample["input_ids"] = list()
                packed_sample["attention_mask"] = list()
                packed_sample["labels"] = list()

            packed_sample["input_ids"].extend(feature["input_ids"])
            packed_sample["attention_mask"].extend(feature["attention_mask"])
            packed_sample["labels"].extend(feature["labels"])
        batched_features.append(copy.deepcopy(packed_sample))

        labels = [feature["labels"] for feature in batched_features]
        no_labels_features = [{k: v for k, v in feature.items() if k != "labels"} for feature in batched_features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        
        if padding_side == "right":
            batch["labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        return batch


def construct_prompt(
    text: str, model_type: str, first_round: bool=False
) -> str:
    if model_type == "codellama":
        prompt = PROMPT_TEMPLATE[model_type].format(text=text)
    elif model_type == "codegemma":
        prompt = PROMPT_TEMPLATE[model_type].format(text=text)
        if not first_round:
            prompt = "<end_of_turn>\n" + prompt
    elif model_type == "deepseek" or model_type == "starcoder2":
        prompt = PROMPT_TEMPLATE[model_type].format(text=text)
        if not first_round:
            prompt = "\n" + prompt
    elif model_type == "codeqwen":
        prompt = PROMPT_TEMPLATE[model_type].format(text=text)
        if not first_round:
            prompt = "\n" + prompt
    elif model_type == "llama3":
        prompt = PROMPT_TEMPLATE[model_type].format(text=text)
        if not first_round:
            prompt = "<|eot_id|>" + prompt
    
    return prompt


def convert_text_to_id(
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str, response: str, model_type: str, label_pad_token_id: int, first_round: bool=False,
):
    if model_type == "codellama":
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    elif model_type == "codegemma":
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=first_round)
    elif model_type == "deepseek" or model_type == "starcoder2":
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=first_round)
    elif model_type == "codeqwen":
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    elif model_type == "llama3":
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=first_round)
    
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    label_ids = [label_pad_token_id] * len(prompt_ids) + response_ids

    return input_ids, label_ids


def preprocess(
    sources: List[List[Dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model_type: str,
    label_pad_token_id: int=IGNORE_TOKEN_ID,
):
    # check model type, support codellama, codegemma and deepseek
    if model_type not in PROMPT_TEMPLATE.keys():
        raise ValueError(
            f"The types of models that can be processed are {list(PROMPT_TEMPLATE.keys())}, but you provided {model_type}"
        )

    batch_input_ids = []
    batch_label_ids = []
    batch_attention_mask = []
    for conversation in sources:
        prompt = response = None
        input_ids = []
        label_ids = []
        attention_mask = []
        for i, turn in enumerate(conversation):
            if turn["role"] == "user":
                prompt = construct_prompt(turn["content"], model_type, first_round=(i < 2))
            elif turn["role"] == "assistant":
                response = turn["content"]
            
            if prompt and response:
                # convert text to ids
                round_input_ids, round_lable_ids = convert_text_to_id(tokenizer, prompt, response, model_type, label_pad_token_id, first_round=(i < 2))
                round_attention_mask = [1] * len(round_input_ids)

                input_ids.extend(round_input_ids)
                label_ids.extend(round_lable_ids)
                attention_mask.extend(round_attention_mask)
                
                prompt = response = None
        
        if len(label_ids) == 0:
            continue
        
        batch_input_ids.append(input_ids)
        batch_label_ids.append(label_ids)
        batch_attention_mask.append(attention_mask)

    return dict(
        input_ids=batch_input_ids,
        labels=batch_label_ids,
        attention_mask=batch_attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: List[Dict[str, Any]], tokenizer: transformers.PreTrainedTokenizerBase, model_type: str):
        super().__init__()
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, i) -> Dict[str, List[int]]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        res = preprocess([self.raw_data[i]["conversation"]], self.tokenizer, self.model_type)
        res = dict(
            input_ids=res["input_ids"][0],
            labels=res["labels"][0],
            attention_mask=res["attention_mask"][0],
        )
        self.cached_data_dict[i] = res
        return res


def create_dataset(
    tokenizer: transformers.PreTrainedTokenizerBase, data_path: str, model_type: str
) -> Dataset:
    print("Loading data...")
    try:
        train_json = json.load(open(data_path, "r"))
    except:
        train_json = []
        with open(data_path, "r") as f:
            for line in f:
                train_json.append(json.loads(line))
    
    train_dataset = SupervisedDataset(train_json, tokenizer, model_type)

    return train_dataset

