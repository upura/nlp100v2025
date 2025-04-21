raw_text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
text = raw_text.replace(".", "").replace(",", "")
ans = [len(w) for w in text.split()]
print(ans)
