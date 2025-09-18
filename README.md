# Finetuning HYDE script

---
## **Local setup**
## 1. Setup
```bash
    pip install -r requirements.txt
```

## 2. Data format
Required columns: `chunk, question`

| chunk | question | 
|-------|----------|
| 1 + 1 = 3 | 1 + 1 = ?|

## 3. Running finetune
python train.py

--- 

## **Kaggle setup**

## 1. Acquiring Hugging face model

Go to [Gemma on Hugging Face](https://huggingface.co/google/gemma-3-270m-it) and accept to terms and agreement.

Creating login token on Hugging face, create API key on Kaggle by going to `Add-ons` -> `Secret`

Login to Hugging face using secret key
```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("hugging-face")

from huggingface_hub import login

login(token=secret_value_0)
```

## 2. Cloning source code 
```bash
!git clone https://github.com/TranDuong874/finetune-hyde.git
````




