import re

import pandas as pd

df = pd.read_json("ch03/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
for file in re.findall(r"\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]", uk_text):
    print(file[1])
