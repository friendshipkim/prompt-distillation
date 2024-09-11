import torch
import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch



def tokenize(example, tokenizer, seq_length, dataset_text_field='text'):

    text = example[dataset_text_field]

    all_tokens = tokenizer(text, return_tensors='pt')
    input_ids = all_tokens['input_ids']
    attn_mask = all_tokens['attention_mask']
    breakpoint()
    
    # Count how many EOS tokens are in input_ids
    eos_token_id = tokenizer.eos_token_id
    eos_count = (input_ids == eos_token_id).sum().item()
    

    loss_mask = attn_mask.clone()
    loss_mask[:, :prompt_len] = 0
    return {
        'input_ids': input_ids[:, :seq_length],
        'labels': input_ids[:, :seq_length],
        'attention_mask': attn_mask[:, :seq_length],
        'loss_mask': loss_mask[:, :seq_length],
    }


def streaming_data_prep(dataset, split, tokenizer, seq_length, min_completion_tokens=3):
    is_train = split == 'train'
    dataset = dataset.map(
        lambda example: tokenize(example, tokenizer, seq_length),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(
        lambda example: example['loss_mask'].sum().item() > min_completion_tokens,
    )
    if is_train:
        dataset = InfiniteDataset(dataset)

    return dataset

class InfiniteDataset(IterableDataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                print('WARNING: The dataset reached end and the iterator is reset to the start.')
                iterator = iter(self.dataset)


class DataCollatorWithPadding:

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def paddingtensor2D(self, in_tensors):
        b, l = in_tensors.shape
        padding_tensor = torch.zeros(b, self.seq_length - l, dtype=in_tensors.dtype)
        out_tensors = torch.cat((in_tensors, padding_tensor), dim=1)
        return out_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids']) for item in features])
        batch_labels = torch.cat([self.paddingtensor2D(item['labels']) for item in features])
        batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask']) for item in features])
        batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask']) for item in features])
        batch = {
            'input_ids': batch_input_ids,
            'labels': batch_labels,
            'attention_mask': batch_attention_mask,
            'loss_mask': batch_loss_mask,
        }
        return batch


def preprocess_datasets(train_dataset, eval_dataset, tokenizer):
    # train_dataset = load_dataset(
    #     'json',
    #     data_files=['/var/cr05_data/avner/data/OpenHermes2.5_prompt_response_train.jsonl'],
    #     split='train',
    #     streaming=True,
    # )
    # eval_dataset = load_dataset(
    #     'json',
    #     data_files=['/var/cr05_data/avner/data/OpenHermes2.5_prompt_response_valid.jsonl'],
    #     split='train',
    #     streaming=True,
    # )
    train_dataset = train_dataset.shuffle(seed=42)
    seq_length = tokenizer.model_max_length
    
    train_dataset = streaming_data_prep(train_dataset, 'train', tokenizer, seq_length, min_completion_tokens=3)
    eval_dataset = streaming_data_prep(eval_dataset, 'val', tokenizer, seq_length, min_completion_tokens=3)

    return train_dataset, eval_dataset