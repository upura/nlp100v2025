import pandas as pd

N = 10
df = pd.read_csv('ch02/popular-names.txt', sep='\t', header=None)
print(df.head(N).to_csv(sep=' ', index=False, header=None))
