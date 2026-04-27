"""Quick test script"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "C:/Users/lians/qwen_model/Qwen/Qwen2___5-3B"
print(f'Model path: {model_path}')

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0')
print('Model loaded successfully!')
