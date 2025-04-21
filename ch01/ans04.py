def extract_chars(i, word):
    if i in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        return (word[0], i)
    else:
        return (word[:2], i)


raw_text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
text = raw_text.replace(".", "").replace(",", "")
ans = [extract_chars(i, w) for i, w in enumerate(text.split(), 1)]
print(dict(ans))
