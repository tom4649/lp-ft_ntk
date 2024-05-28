import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data, clean
set_seed(0)

import os
import re
import random
import numpy as np
import pandas as pd

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {1:0, 2:1}
def label_process(label):
    label = label_mapping[label]
    return label



def read_data(path):
    dataset = pd.read_csv(path, sep="\t", header=0)
    dataset = dataset[["Post Text", "Hate/Non-Hate"]].rename(columns={"Post Text":"text", "Hate/Non-Hate":"label"})
    dataset["text"] = dataset["text"].apply(clean).apply(text_process)
    dataset["label"] = dataset["label"].apply(label_process)
    dataset = [{"text":data[0], "label":data[1]} for data in dataset.values]
    return dataset

# load and shuffle
abuse_analyzer = read_data("./datasets/raw/ToxicDetection/abuse_analyzer/AbuseAnalyzer_Dataset.tsv")
random.shuffle(abuse_analyzer)

# split
test = abuse_analyzer

# test
test_dataset = []
for data in test:
    test_dataset.append((data["text"], data["label"]))

# save
save_data(test_dataset, "./datasets/process/ToxicDetection/abuse_analyzer", "test")
