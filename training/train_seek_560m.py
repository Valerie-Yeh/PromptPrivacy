from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator, BitsAndBytesConfig
import torch
import pandas as pd
from datasets import Dataset
import os
import numpy as np
from typing import Dict

torch.cuda.empty_cache()

deepspeed_config_file = 'ds_config_zero3.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'bigscience/bloomz-560m'
max_len = 800
batch_size = 2

# load dataset
df_en = pd.read_csv('../dataset_43k/train.csv', encoding='utf-8')
df_en = df_en.drop(columns=['outputs'])
df_zh = pd.read_csv('../dataset_zh/train.csv', encoding='utf-8')
df = pd.concat([df_en, df_zh])
df = df.drop(columns=['id', 'given_words'])

#input = df[['anonymized_inputs', 'anonymized_outputs', 'inputs']]
#output = df['gt']

#data_train = pd.merge(input, output, left_index=True, right_index=True)
data_train = Dataset.from_pandas(df, preserve_index=False)


# set up AutoModelForCausalLM
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    load_in_4bit = False,
    llm_int8_threshold = 6.0,
    llm_int8_skip_modules = None,
    llm_int8_enable_fp32_cpu_offload = False,
    llm_int8_has_fp16_weight = False,
    bnb_4bit_quant_type = 'fp4',
    bnb_4bit_use_double_quant = False,
    bnb_4bit_compute_dtype = torch.float32,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    torch_dtype = torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


prefix = "Input: "
translation = "\n\nGeneration: "
suffix = "\n\nInput: "

def preprocess_function(examples):
    inputs = []
    for i in range(len(examples["anonymized_inputs"])):
        inputs.append(prefix + examples["anonymized_inputs"][i] + translation + examples["anonymized_outputs"][i] + suffix + examples["inputs"][i] + translation + tokenizer.bos_token + examples["gt"][i] + tokenizer.eos_token)
    
    model_inputs = tokenizer(inputs, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

train_dataset = data_train.map(preprocess_function, batched=True, remove_columns=data_train.column_names)


# Create a configuration (LoraConfig) where I define LoRA-specific parameters.
lora_config = LoraConfig(
    r = 32,
    lora_alpha = 16, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules = ["query_key_value"],
    lora_dropout = 0.05, 
    bias = "none", # this specifies if the bias parameter should be trained. 
    task_type = "CAUSAL_LM"
)

# Wrap the base model with get_peft_model() to get a trainable PeftModel.
peft_model = get_peft_model(model, lora_config)
print(peft_model.print_trainable_parameters())


output_directory = os.path.join("../cache_gen_bi", "seek_outputs")
training_args = TrainingArguments(
    report_to = "none",
    output_dir = output_directory,
    #evaluation_strategy = 'epoch',
    per_device_train_batch_size = batch_size,
    #per_device_eval_batch_size = batch_size,
    #eval_accumulation_steps = 1,
    learning_rate =  2e-4, # Higher learning rate than full fine-tuning.
    num_train_epochs = 10,
    save_strategy = 'epoch',
    deepspeed = deepspeed_config_file,
)

trainer = Trainer(
    model = peft_model.to(device),
    args = training_args,
    train_dataset = train_dataset,
    #eval_dataset = test_dataset,
    tokenizer = tokenizer,
    data_collator = default_data_collator,
    #compute_metrics = compute_metrics,
)
trainer.train()

peft_model_path = os.path.join(output_directory, 'model_560m')
trainer.model.save_pretrained(peft_model_path)