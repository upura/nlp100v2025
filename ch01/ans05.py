def n_gram(target, n):
    return [target[idx : idx + n] for idx in range(len(target) - n + 1)]


text = "I am an NLPer"
for i in range(1, 4):
    print(n_gram(text, i))
    print(n_gram(text.split(" "), i))
