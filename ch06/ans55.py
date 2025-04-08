import pandas as pd

df = pd.read_csv("ch06/ans54.txt", sep=" ", header=None)
print((df[3] == df[4]).sum() / len(df))
