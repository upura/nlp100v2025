import pandas as pd


df_train = pd.read_csv("ch07/SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

print(df_train["label"].value_counts())
print(df_dev["label"].value_counts())
