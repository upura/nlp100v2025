import pandas as pd

df = pd.read_csv("ch02/popular-names.txt", sep="\t", header=None)
print(df[0])
