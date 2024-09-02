import os
import gc
import time
from typing import Dict
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model_dir = 'bigscience/bloomz-560m'

# specify tast type and langauge
task_type = 'gen'
lang = 'en'

seek_model_path = f"../cache_gen_bi/seek_outputs/model_560m"

# load dataset
df = pd.read_csv('../dataset_bi/test.csv', encoding='utf-8')

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    load_in_4bit = False,
    llm_int8_threshold = 6.0,
    llm_int8_skip_modules = None,
    llm_int8_enable_fp32_cpu_offload = False,
    llm_int8_has_fp16_weight = False,
    bnb_4bit_quant_type = 'fp4',
    bnb_4bit_use_double_quant = False,
    bnb_4bit_compute_dtype = torch.float16,
)


def recover_text(model, tokenizer, task_type, lang):
    re_model = PeftModel.from_pretrained(model, seek_model_path, quantization_config=bnb_config, device_map='cuda:0')
    with open(f'../prompts/seek_{task_type}_{lang}.txt', 'r', encoding='utf-8') as f:
        initial_prompt = f.read()

    
    for i in tqdm(range(len(df['inputs']))):
        test_targets = []
        test_targets.append(i)
        input_text = initial_prompt % (df["anonymized_inputs"][i], df["anonymized_outputs"][i], df["inputs"][i])
        input_text += tokenizer.bos_token
        inputs = tokenizer(input_text, return_tensors='pt')
        inputs = inputs.to('cuda:0')
        len_prompt = len(inputs['input_ids'][0])
        def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
            cur_top1 = tokenizer.decode(input_ids[0,len_prompt:])
            if '\n' in cur_top1 or tokenizer.eos_token in cur_top1:
                return True
            return False
        pred = re_model.generate(
            **inputs,
            generation_config = GenerationConfig(
                max_length=400,
                do_sample=False,
                num_beams=3,
                ),
            stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
            )
        pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
        recovered_text = tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
        test_targets.append(recovered_text)

        with open('../dataset_bi/predictions_560m.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(test_targets)


if __name__ == '__main__':
    # load models
    print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, quantization_config=bnb_config, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    field = ['id', 'outputs']
    with open('../dataset_bi/predictions_560m.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(field)
    recover_text(model, tokenizer, task_type, lang)