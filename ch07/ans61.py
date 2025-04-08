import pandas as pd
from collections import defaultdict


def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data


df_train = pd.read_csv("ch07/SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

data_train = []
for sentence, label in zip(df_train["sentence"], df_train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))

print(data_train[0])
