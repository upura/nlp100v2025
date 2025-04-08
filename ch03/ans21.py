import pandas as pd

df = pd.read_json("ch03/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_texts = uk_text.split("\n")
ans = list(filter(lambda x: "[Category:" in x, uk_texts))
print(ans)
