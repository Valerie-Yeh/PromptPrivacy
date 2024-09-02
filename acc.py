# Compute the accuracy of seek model output

import pandas as pd
import re
from statistics import mean

pred = pd.read_csv("dataset_43k/predictions_560m.csv")
gt = pd.read_csv("dataset_43k/test.csv")

def normalize_pattern(pattern):
    # Remove spaces around special characters
    pattern = re.sub(r'\s*:\s*', ':', pattern)
    pattern = re.sub(r'\s*/\s*', '/', pattern)
    pattern = re.sub(r'\s*-\s*', '-', pattern)
    pattern = re.sub(r'\s*\.\s*', '.', pattern)
    pattern = re.sub(r'\s*@\s*', '@', pattern)
    return pattern

def count_occurrence(text, items):
    counts = {}
    for item in items:
        pattern = re.compile(r'\b' + re.escape(item) + r'\b')
        counts[item] = len(pattern.findall(text))
    return counts


if __name__ == "__main__":
    accuracies = []
    for i in range(len(pred['outputs'])):
        pred_output = pred['outputs'][i].lower()
        pred_output = normalize_pattern(pred_output)
        ano_output = gt['anonymized_outputs'][i]
        ano_output = normalize_pattern(ano_output)
        tag = gt['given_words'][i]
        if tag[0] == "'":
            tag = tag[1:-1]
        tag = tag.split("', '")
        
        key_mapping = {}
        label_split = []
        ent_label = []
        for element in tag:
            k = element.split(': ', 1)[0]
            k = normalize_pattern(k)
            v = element.split(': ', 1)[1]
            v = normalize_pattern(v)
            label_split.append(k)
            ent_label.append(v)
            key_mapping[k] = v

        #counts = count_occurrence(pred_output, label_split)
        counts = count_occurrence(pred_output, ent_label)
        matches = count_occurrence(ano_output, label_split)
        """
        accuracy = []
        for key, value in matches.items():
            if counts[key] != 0:
                accuracy.append(0)
            else:
                accuracy.append(1)

        accuracies.append(mean(accuracy))
        """
        for key, value in matches.items():
            if counts[key_mapping[key]] > value:
                counts[key_mapping[key]] = value
        
        if sum(list(matches.values())) == 0:
            accuracies.append(1)
        else:
            accuracies.append(sum(list(counts.values())) / sum(list(matches.values())))
        
    print(mean(accuracies))