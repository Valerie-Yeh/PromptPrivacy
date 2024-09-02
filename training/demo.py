from typing import Dict, List
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import openai

base_model_dir = 'bigscience/bloomz-560m'
seek_model_path = f"../cache_gen/seek_outputs/model"

# specify langauge
task_type = 'gen'
lang = 'en'

hide_pii = pipeline("token-classification", model="Isotonic/distilbert_finetuned_ai4privacy")

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

def merge_pii(raw_input, input_dicts: List[Dict]) -> List[Dict]:
    discarded_types = ['FIRSTNAME', 'MIDDLENAME', 'LASTNAME', 'FULLNAME', 'STATE', 'COUNTY', 'CITY']
    entities = []
    prev_label = None
    for input_dict in input_dicts:
        ent = input_dict['entity']
        start = input_dict['start']
        end = input_dict['end']

        bio = ent[:1]
        assert bio in ['B', 'I']
        ent_name = ent[2:]
        if bio == 'B':
            entities.append({
                "start": start,
                "end": end,
                "ent": ent_name,
            })
            prev_label = ent_name
        else:
            assert prev_label == ent_name
            entities[-1]['end'] = end
    
    output_dicts = []
    ent_count = defaultdict(int)
    #print(entities)
    for entity in entities:
        full_ent = entity['ent']
        start_idx = entity['start']
        end_idx = entity['end']
        ent_count[full_ent] += 1
        if full_ent not in discarded_types:
            word = raw_input[start_idx:end_idx]
            output_dict = {
                '{}_{}'.format(full_ent, str(ent_count[full_ent])): word
            }
            output_dicts.append(output_dict)

    return output_dicts


def get_anonymized_output(raw_input, given_words):
    for given_word in given_words:
        if list(given_word.values())[0] in raw_input:
            raw_input = raw_input.replace(list(given_word.values())[0], '[{}]'.format(list(given_word.keys())[0]))
    return raw_input

def get_api_output(anonymized_input, given_words):
    response = openai.chat.completions.create(
        model = "gpt-4",
        temperature = 0.1,
        messages=[
            {"role": "system", "content": "Generate within 200 words. And then in generated text, replace uncased words in the Given words with cased words staring with '[' and ending with ']' such as [FULLNAME_1], [URL_1]. For example, Text: In our video conference, discuss the role of evidence in the arbitration process involving [FULLNAME_1] and [FULLNAME_2].\n\nGiven words: 'FULLNAME_1: dr. marvin rolfson', 'FULLNAME_2: julius daugherty'\n\nAnonymized output: Evidence plays a critical role in arbitral proceedings, providing the basis for evaluating each party's claims and defenses. Through documents, testimonies, and other forms of evidence, [FULLNAME_1] and [FULLNAME_2] present their arguments and support their positions."},
            {"role": "user", "content": f"Text: {anonymized_input}\n\nGiven words: {given_words}\n\nAnonymized output:"}
            ]
        )
    result = response.choices[0].message.content.strip(" \n")
    result = result.replace('\n', ' ')

    return result

def recover_text(anonymized_input, anonymized_output, raw_input, model, tokenizer):
    re_model = PeftModel.from_pretrained(model, seek_model_path, quantization_config=bnb_config, device_map='cuda:0')
    with open(f'../prompts/seek_{task_type}_{lang}.txt', 'r', encoding='utf-8') as f:
        initial_prompt = f.read()

    input_text = initial_prompt % (anonymized_input, anonymized_output, raw_input)
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
    return recovered_text

if __name__ == '__main__':
    # load models
    print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, quantization_config=bnb_config, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    while True:
        # input text
        raw_input = input('\033[1;31mInput:\033[0m ')
        if raw_input == 'q':
            print('quit')
            break
        # hide
        hidden_list = hide_pii(raw_input)
        given_words = merge_pii(raw_input, hidden_list)
        anonymized_input = get_anonymized_output(raw_input, given_words)
        print('\033[1;31mHidden Input:\033[0m ', anonymized_input)
        # seek
        anonymized_output = get_api_output(anonymized_input, given_words)
        print(f'\033[1;31mHidden Output:\033[0m ', anonymized_output)
        output_text = recover_text(anonymized_input, anonymized_output, raw_input, model, tokenizer)
        print(f'\033[1;31mRecovered Output:\033[0m ', output_text)