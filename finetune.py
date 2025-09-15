import pandas as pd
from datasets import load_dataset
import optuna
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import numpy as np
import yaml
import torch

def create_conversation(sample):
    return {
        "messages" : [
            {"role" : "user", "content" : sample["questions"]},
            {"role" : "assistant", "content" : sample["chunk"]}
        ]
    }

def load_data(data_path, test_size=0.2):
    dataset = load_dataset('csv', data_files=data_path, split='train')
    # dataset = dataset.select(range(100)) # sanity check
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
    return dataset

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    MODEL_NAME = config['training']['model']
    DATA_PATH = config['dataset']['data_path']
    
    model, tokenizer = load_model(MODEL_NAME)
    dataset = load_dataset(DATA_PATH)

    args = SFTConfig(
        output_dir=config["training"]["output_dir"],
        max_seq_length=config["training"]["max_length"],
        packing=config["training"]["packing"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim=config["training"]["optim"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        learning_rate=config["training"]["learning_rate"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        push_to_hub=config["training"]["push_to_hub"],
        report_to=config["training"]["report_to"],
        dataset_kwargs=config["dataset"]["dataset_kwargs"],
        fsdp=config["training"]["fsdp"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )
