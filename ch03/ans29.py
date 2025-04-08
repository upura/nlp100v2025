import re

import pandas as pd
import requests


def remove_stress(dc):
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}


def remove_inner_links(dc):
    r = re.compile("\[\[(.+\||)(.+?)\]\]")
    return {k: r.sub(r"\2", v) for k, v in dc.items()}


def remove_mk(v):
    r1 = re.compile("'+")
    r2 = re.compile("\[\[(.+\||)(.+?)\]\]")
    r3 = re.compile("\{\{(.+\||)(.+?)\}\}")
    r4 = re.compile("<\s*?/*?\s*?br\s*?/*?\s*>")
    v = r1.sub("", v)
    v = r2.sub(r"\2", v)
    v = r3.sub(r"\2", v)
    v = r4.sub("", v)
    return v


def get_url(dc):
    url_file = dc["国旗画像"].replace(" ", "_")
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&titles=File:"
        + url_file
        + "&prop=imageinfo&iiprop=url&format=json"
    )
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)


df = pd.read_json("ch03/jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
uk_texts = uk_text.split("\n")

pattern = re.compile("\|(.+?)\s=\s*(.+)")
ans = {}
for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        ans[r[1]] = r[2]

r = re.compile("\[\[(.+\||)(.+?)\]\]")
ans = {k: r.sub(r"\2", remove_mk(v)) for k, v in ans.items()}
print(get_url(remove_inner_links(remove_stress(ans))))
