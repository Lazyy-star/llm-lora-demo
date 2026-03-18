from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

data = [
    {"text": "Q: What is AI?\nA: Artificial Intelligence is a field of study..."},
    {"text": "Q: What is machine learning?\nA: Machine learning is a subset of AI..."}
]

dataset = Dataset.from_list(data)

def tokenize(example):
    result = tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
    result["labels"] = result["input_ids"].copy()
    return result
dataset = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./lora_model")