#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Union
from functools import partial

import torch
import transformers
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training
from datasets import load_dataset


PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


# NOTE: We can't use 3.10's new X|Y syntax b/c HfArgumentParser doesn't support it.
# https://github.com/huggingface/transformers/issues/20249
@dataclass
class ModelArguments:
    model_name_or_path: str
    device_map: Union[None, str, dict[str, Union[int, str, torch.device]]] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data_cleaned.json", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_lora_8bit: bool = field(default=False)


@dataclass
class LoRAArguments:
    lora_r: int = field(default=16, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


def generate_prompt(example: dict[str, str]) -> dict[str, str]:
    if example.get("input", "") == "":
        prompt = PROMPT_NO_INPUT.format_map(example)
    else:
        prompt = PROMPT_INPUT.format_map(example)
    example["prompt"] = prompt
    return example


def batch_tokenize(
    tokenizer: transformers.PreTrainedTokenizer, example: dict[str, list[str]]
) -> dict[str, list[list[int]]]:
    # append an eos token and tokenize.
    # since examples will be batched by the collator, we don't need to generate attention masks here.
    # labels will also be generated by the collator (just a copy of input_ids)
    tokenized_prompt = tokenizer(example["prompt"], return_attention_mask=False)
    for input_ids in tokenized_prompt["input_ids"]:
        input_ids.append(tokenizer.eos_token_id)
    return tokenized_prompt


def train() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    lora_args: LoRAArguments
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )
    if tokenizer.pad_token is None:
        # LLaMA tokenizer doesn't have a pad token by default, so we just use the unk token to pad.
        # This is preferable to adding a pad token as this will keep the vocabulary size a multiple of 8.
        # Note that we use the unk token instead of the eos token b/c we don't want DataCollatorForLanguageModeling
        # to use -100 for the eos token.
        tokenizer.pad_token = tokenizer.unk_token

    if training_args.use_lora_8bit:
        logging.warning("Using LoRA and 8-bit training")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, load_in_8bit=True, device_map=model_args.device_map
        )
        # use_gradient_checkpointing=False since it doesn't play with torch.compile()
        # https://github.com/pytorch/pytorch/issues/97077
        # https://github.com/pytorch/pytorch/issues/97436
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
            ),
        )
        model.print_trainable_parameters()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, device_map=model_args.device_map
        )

    dataset = (
        load_dataset("json", data_files=data_args.data_path)
        .map(generate_prompt, remove_columns=["instruction", "input", "output"])
        .map(partial(batch_tokenize, tokenizer), batched=True, remove_columns="prompt")
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        ),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
