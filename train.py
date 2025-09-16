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
from contextlib import contextmanager

def create_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["chunk"]}
        ]
    }

def load_data(data_path, test_size=0.2):
    dataset = load_dataset('csv', data_files=data_path, split='train')
    # dataset = dataset.select(range(100))  # sanity check
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
    return dataset

@contextmanager
def model_context(model_name):
    """Context manager to ensure proper model cleanup"""
    model = None
    tokenizer = None
    try:
        print(f"Loading model: {model_name}")
        # Load with memory-efficient settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            attn_implementation='eager',
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        yield model, tokenizer
    finally:
        # Ensure cleanup happens even if there's an exception
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Model cleaned up from memory")

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Additional CUDA cleanup
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass

def objective(trial, model_name, dataset, config, temp_dir):
    """Objective function for Optuna optimization - loads model fresh each time"""
    print(f"\nStarting trial {trial.number}")
    aggressive_cleanup()
    
    hp_space = config["hyperparameter_space"]

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 
        float(hp_space['learning_rate']['min']), 
        float(hp_space['learning_rate']['max']), 
        log=hp_space['learning_rate']['log']
    )

    optimizer = trial.suggest_categorical("optimizer", 
        hp_space['optimizer']['choices']
    )
    
    num_train_epochs = trial.suggest_int("num_train_epochs", 
        hp_space['num_train_epoch']['min'],
        hp_space['num_train_epoch']['max']
    )

    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", 
        hp_space['per_device_train_batch_size']['choices']
    )
    
    # Create unique output directory for this trial
    trial_output_dir = os.path.join(temp_dir, f'trial_{trial.number}')
    os.makedirs(trial_output_dir, exist_ok=True)
    
    try:
        # Load model fresh for each trial to avoid memory accumulation
        with model_context(model_name) as (model, tokenizer):
            # Reduce dataset size for hyperparameter search (optional)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
            
            # Optional: Use smaller subset for faster hyperparameter search
            if config.get("optuna", {}).get("use_subset", False):
                subset_size = config["optuna"].get("subset_size", 1000)
                if len(train_dataset) > subset_size:
                    train_dataset = train_dataset.select(range(subset_size))
                if len(eval_dataset) > subset_size // 4:
                    eval_dataset = eval_dataset.select(range(subset_size // 4))
            
            # Memory-optimized training config
            args = SFTConfig(
                output_dir=trial_output_dir,
                max_length=min(config["training"]["max_length"], 512),  # Reduce sequence length
                packing=config["training"]["packing"],
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=max(1, per_device_train_batch_size // 2),  # Smaller eval batch
                gradient_checkpointing=True,  # Always use for memory savings
                gradient_accumulation_steps=max(1, 8 // per_device_train_batch_size),  # Maintain effective batch size
                optim=optimizer,
                logging_steps=config["training"]["logging_steps"],
                save_strategy="no",  # Don't save during hyperparameter search
                eval_strategy="epoch",  # Evaluate less frequently
                eval_steps=None,  # Only at end of epochs
                learning_rate=learning_rate,
                fp16=config["training"]["fp16"],
                bf16=config["training"]["bf16"],
                lr_scheduler_type=config["training"]["lr_scheduler_type"],
                push_to_hub=False,
                report_to="none",
                dataset_kwargs=config["dataset"]["dataset_kwargs"],
                fsdp=config["training"]["fsdp"],
                dataloader_pin_memory=False,  # Disable pin memory to save RAM
                dataloader_num_workers=0,  # Disable multiprocessing
                remove_unused_columns=True,
                prediction_loss_only=True,  # Only compute loss, not other metrics
                max_grad_norm=1.0,  # Prevent gradient explosion
                warmup_ratio=0.1,  # Shorter warmup
            )
            
            # Create trainer
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            
            # Train the model
            result = trainer.train()
            
            # Quick evaluation - just get the loss
            eval_results = trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            
            print(f"Trial {trial.number} completed - Eval loss: {eval_loss:.4f}")
            
            # Clean up trainer before returning
            del trainer
            del result
            
            return eval_loss
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')
    
    finally:
        # Clean up trial directory to save disk space
        if os.path.exists(trial_output_dir):
            try:
                shutil.rmtree(trial_output_dir)
            except:
                pass
        aggressive_cleanup()

def run_hyperparameter_optimization(model_name, dataset, config, n_trials=6):
    """Run Optuna hyperparameter optimization with memory management"""
    print("Starting hyperparameter optimization...")
    
    # Create temporary directory for trials
    temp_dir = os.path.join(config["training"]["output_dir"], "optuna_trials")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create Optuna study with pruning for failed trials
    study = optuna.create_study(
        direction='minimize', 
        sampler=GPSampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5)
    )
    
    # Optimize with proper error handling
    try:
        study.optimize(
            lambda trial: objective(trial, model_name, dataset, config, temp_dir), 
            n_trials=n_trials,
            catch=(Exception,)  # Catch exceptions and continue
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Get best parameters
    if len(study.trials) > 0:
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\nHyperparameter optimization completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best eval loss: {best_value:.4f}")
        print(f"Number of successful trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        return best_params
    else:
        print("No successful trials completed!")
        return None

def train_final_model(model_name, dataset, config, best_params):
    """Train final model with best hyperparameters"""
    print("\nTraining final model with best hyperparameters...")
    
    aggressive_cleanup()
    
    if best_params is None:
        print("No best parameters found, using default configuration")
        best_params = {}
    
    with model_context(model_name) as (model, tokenizer):
        # Create final SFTConfig with best parameters
        args = SFTConfig(
            output_dir=config["training"]["output_dir"],
            max_length=config["training"]["max_length"],
            packing=config["training"]["packing"],
            num_train_epochs=best_params.get('num_train_epochs', config["training"]["num_train_epochs"]),
            per_device_train_batch_size=best_params.get('per_device_train_batch_size', config["training"]["per_device_train_batch_size"]),
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
            optim=best_params.get('optimizer', config["training"].get("optim", "adamw_torch")),
            logging_steps=config["training"]["logging_steps"],
            save_strategy=config["training"]["save_strategy"],
            eval_strategy=config["training"]["eval_strategy"],
            learning_rate=best_params.get('learning_rate', config["training"]["learning_rate"]),
            fp16=config["training"]["fp16"],
            bf16=config["training"]["bf16"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            push_to_hub=config["training"]["push_to_hub"],
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
    
    # Load data (but not model yet - we'll load it fresh for each trial)
    print("Loading data...")
    dataset = load_data(DATA_PATH)
    
    # Run hyperparameter optimization
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 6)
    
    best_params = run_hyperparameter_optimization(
        MODEL_NAME, dataset, config, n_trials=n_trials
    )
    
    # Train final model with best parameters
    if best_params is not None:
        final_trainer = train_final_model(MODEL_NAME, dataset, config, best_params)
        print("\nTraining completed successfully!")
        print(f"Best hyperparameters used: {best_params}")
    else:
        print("Hyperparameter optimization failed, training with default parameters")
        final_trainer = train_final_model(MODEL_NAME, dataset, config, {})