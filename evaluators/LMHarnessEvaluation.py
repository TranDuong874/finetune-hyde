from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import os, json
import torch
import numpy as np

import copy

class LMHarnessEvaluation:
    def __init__(self, cfg, model=None, tokenizer=None):
        """
        cfg: dict like your example
        model: optional in-memory HuggingFace model
        tokenizer: required if model is in-memory
        """
        self.cfg = cfg
        self.model_obj = model
        self.tokenizer = tokenizer

        self.model_name_or_obj = cfg.get("model_name_or_obj")
        self.device = cfg.get("device", "cuda")
        self.output_dir = cfg.get("output_dir", "results")
        self.task_names = cfg.get("task_names", ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc2"])
        self.num_fewshot = cfg.get("num_fewshot", 5)
        self.batch_size = cfg.get("batch_size", "auto")
        self.limit = cfg.get("limit", None)
        self.apply_chat_template = cfg.get("apply_chat_template", True)
        self.fewshot_as_multiturn = cfg.get("fewshot_as_multiturn", True)
        self.system_instruction = cfg.get("system_instruction", "")

        os.makedirs(self.output_dir, exist_ok=True)
        self.task_manager = tasks.get_task_dict(self.task_names)



    def make_serializable(self, d):
        d = copy.deepcopy(d)
        # remove all 'process_results', 'process_docs', etc.
        for task in d.get("configs", {}):
            for key in list(d["configs"][task].keys()):
                if callable(d["configs"][task][key]):
                    del d["configs"][task][key]
        return d
    
    def convert_objs(self, o):
        import numpy as np
        import torch
        
        # Handle None
        if o is None:
            return None
        
        # PyTorch tensors (all types)
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().numpy().tolist()
        
        # PyTorch dtypes
        if isinstance(o, torch.dtype):
            return str(o)
        
        # PyTorch device
        if isinstance(o, torch.device):
            return str(o)
        
        # PyTorch memory format
        if hasattr(torch, 'memory_format') and isinstance(o, torch.memory_format):
            return str(o)
        
        # NumPy arrays (all types including structured arrays)
        if isinstance(o, np.ndarray):
            if o.dtype.names:  # structured array
                return {name: self.convert_objs(o[name].tolist()) for name in o.dtype.names}
            else:
                return o.tolist()
        
        # NumPy scalar types (all of them)
        if isinstance(o, np.generic):
            return o.item()
        
        # NumPy dtypes
        if isinstance(o, np.dtype):
            return str(o)
        
        # NumPy matrix (deprecated but still exists)
        if isinstance(o, np.matrix):
            return o.tolist()
        
        # NumPy masked arrays
        if isinstance(o, np.ma.MaskedArray):
            return {
                'data': o.data.tolist(),
                'mask': o.mask.tolist() if hasattr(o.mask, 'tolist') else o.mask,
                'fill_value': self.convert_objs(o.fill_value)
            }
        
        # Handle special float values
        if isinstance(o, float):
            if np.isnan(o):
                return "NaN"
            elif np.isinf(o):
                return "Infinity" if o > 0 else "-Infinity"
            else:
                return o
        
        # Handle complex numbers
        if isinstance(o, complex):
            return {"real": self.convert_objs(o.real), "imag": self.convert_objs(o.imag)}
        
        # Handle bytes
        if isinstance(o, bytes):
            return o.decode('utf-8', errors='replace')
        
        # Handle sets (convert to list)
        if isinstance(o, set):
            return [self.convert_objs(item) for item in sorted(o, key=lambda x: str(x))]
        
        # Handle tuples
        if isinstance(o, tuple):
            return [self.convert_objs(item) for item in o]
        
        # Handle dictionaries
        if isinstance(o, dict):
            return {str(k): self.convert_objs(v) for k, v in o.items()}
        
        # Handle lists
        if isinstance(o, list):
            return [self.convert_objs(item) for item in o]
        
        # Handle other iterables (but not strings)
        if hasattr(o, '__iter__') and not isinstance(o, (str, bytes)):
            try:
                return [self.convert_objs(item) for item in o]
            except TypeError:
                # If iteration fails, convert to string
                return str(o)
        
        # Handle callable objects (functions, methods, etc.)
        if callable(o):
            return f"<callable: {getattr(o, '__name__', str(o))}>"
        
        # Handle objects with custom __dict__
        if hasattr(o, '__dict__') and not isinstance(o, type):
            try:
                return {k: self.convert_objs(v) for k, v in o.__dict__.items() 
                    if not k.startswith('_')}
            except:
                return str(o)
        
        # For any other type, try to convert to string
        try:
            # Try to see if it's JSON serializable first
            import json
            json.dumps(o)
            return o
        except (TypeError, ValueError):
            # If not serializable, convert to string
            return str(o)

        
    def eval(self, output_filename=None):
        """Run evaluation on either model name or in-memory model"""
        # Wrap in-memory model if provided
        if self.model_obj is not None:
            lm = HFLM(
                pretrained=self.model_obj,
                tokenizer=self.tokenizer,
                device=self.device
            )
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=self.task_names,
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )
        else:
            # Load by model name
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.model_name_or_obj},trust_remote_code=True",
                tasks=self.task_names,
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )

        print("="*50)
        print(results)
        print(type(results))
        print("="*50)

        serializable_results = self.make_serializable(results)
        json_ready = self.convert_objs(serializable_results)

        with open("lm_harness_output.json", "w") as f:
            json.dump(serializable_results, f, indent=2)

        # Save results
        if output_filename:
            path = os.path.join(self.output_dir, output_filename)
            with open(path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to {path}")
        return results
    
