import pandas as pd

df = pd.read_csv("ch02/popular-names.txt", sep="\t", header=None)
print(df.sort_values(2, ascending=False))
