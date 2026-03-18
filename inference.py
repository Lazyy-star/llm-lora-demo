from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, "./lora_model")

input_text = "Q: What is AI?\nA:"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0]))