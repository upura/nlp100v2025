import random


def shuffle_word(word):
    if len(word) <= 4:
        return word
    else:
        start = word[0]
        end = word[-1]
        others = random.sample(list(word[1:-1]), len(word[1:-1]))
        return "".join([start] + others + [end])


text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
ans = [shuffle_word(w) for w in text.split()]
print(" ".join(ans))
