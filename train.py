import pandas as pd
from datasets import load_dataset
import optuna
from optuna.samplers import GPSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import numpy as np
import yaml
import torch
import os
import shutil
import gc

def create_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["questions"]},
            {"role": "assistant", "content": sample["chunk"]}
        ]
    }

def load_data(data_path, test_size=0.2):
    dataset = load_dataset('csv', data_files=data_path, split='train')
    # dataset = dataset.select(range(100))  # sanity check
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
    return dataset

def load_model(model_name):
    # Add attn_implementation='eager' for Gemma models as recommended
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def objective(trial, model, tokenizer, dataset, config, temp_dir):
    """Objective function for Optuna optimization"""
    # Clear GPU memory at start of each trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    hp_space = config["hyperparameter_space"]

    # Suggest hyperparameters
    learning_rate = trial.suggest_float( "learning_rate",
        hp_space["learning_rate"]["min"], 
        hp_space["learning_rate"]["max"], 
        log=hp_space["learning_rate"].get("log", False)
    )

    optimizer = trial.suggest_categorical("optimizer", 
        hp_space["optimizer"]["choices"]
    )
    
    num_train_epochs = trial.suggest_int("num_train_epochs", 
        hp_space["per_device_train_batch_size"]["low"],
        hp_space["per_device_train_batch_size"]["high"]
    )

    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", 
        hp_space["num_train_epochs"]["choices"]                                                       
    )
    
    # Create unique output directory for this trial
    trial_output_dir = os.path.join(temp_dir, f'trial_{trial.number}')
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Update SFTConfig with trial parameters
    args = SFTConfig(
        output_dir=trial_output_dir,
        max_length=config["training"]["max_length"],
        packing=config["training"]["packing"],
        num_train_epochs=num_train_epochs,  # or use suggested value
        per_device_train_batch_size=per_device_train_batch_size,  # or use suggested value
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim=optimizer,  # Use suggested optimizer
        logging_steps=config["training"]["logging_steps"],
        save_strategy="no",  # Don't save during hyperparameter search
        eval_strategy=config["training"]["eval_strategy"],
        learning_rate=learning_rate,  # Use suggested learning rate
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        push_to_hub=False,  # Disable during optimization
        report_to="none",  # Disable reporting during optimization
        dataset_kwargs=config["dataset"]["dataset_kwargs"],
        fsdp=config["training"]["fsdp"],
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )
    
    try:
        # Train the model
        trainer.train()
        
        # Evaluate and return the metric to optimize
        eval_results = trainer.evaluate()
        eval_loss = eval_results['eval_loss']
        
        # Clean up trainer and clear memory
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Clean up trial directory to save space
        if os.path.exists(trial_output_dir):
            shutil.rmtree(trial_output_dir)
        
        return eval_loss
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Clean up on failure
        try:
            del trainer
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(trial_output_dir):
            shutil.rmtree(trial_output_dir)
        # Return a high loss value to indicate failure
        return float('inf')

def run_hyperparameter_optimization(model, tokenizer, dataset, config, n_trials=6):
    """Run Optuna hyperparameter optimization"""
    print("Starting hyperparameter optimization...")
    
    # Create temporary directory for trials
    temp_dir = os.path.join(config["training"]["output_dir"], "optuna_trials")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize', sampler=GPSampler())
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, model, tokenizer, dataset, config, temp_dir), 
        n_trials=n_trials
    )
    
    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nHyperparameter optimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best eval loss: {best_value:.4f}")
    
    return best_params

def train_final_model(model, tokenizer, dataset, config, best_params):
    """Train final model with best hyperparameters"""
    print("\nTraining final model with best hyperparameters...")
    
    # Clear memory before final training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create final SFTConfig with best parameters
    args = SFTConfig(
        output_dir=config["training"]["output_dir"],  # Use original output dir
        max_length=config["training"]["max_length"],
        packing=config["training"]["packing"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim=best_params['optimizer'],  # Use best optimizer
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        learning_rate=best_params['learning_rate'],  # Use best learning rate
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        push_to_hub=config["training"]["push_to_hub"],  # Re-enable if needed
        report_to=config["training"]["report_to"],
        dataset_kwargs=config["dataset"]["dataset_kwargs"],
        fsdp=config["training"]["fsdp"],
    )
    
    # Create final trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
    )
    
    # Train final model
    trainer.train()
    
    # Final evaluation
    final_eval_results = trainer.evaluate()
    print(f"Final model evaluation results: {final_eval_results}")
    
    # Save the final model
    trainer.save_model()
    print(f"Final model saved to: {config['training']['output_dir']}")
    
    return trainer

if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    MODEL_NAME = config['training']['model']
    DATA_PATH = config['dataset']['data_path']
    
    # Load model and data
    print("Loading model and data...")
    model, tokenizer = load_model(MODEL_NAME)
    dataset = load_data(DATA_PATH)
    
    # Run hyperparameter optimization
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 6)
    
    best_params = run_hyperparameter_optimization(
        model, tokenizer, dataset, config, n_trials=n_trials
    )
    
    # Train final model with best parameters
    final_trainer = train_final_model(model, tokenizer, dataset, config, best_params)
    
    print("\nTraining completed successfully!")
    print(f"Best hyperparameters used: {best_params}")