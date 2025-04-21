def n_gram(target, n):
    return [target[idx : idx + n] for idx in range(len(target) - n + 1)]


X_text = "paraparaparadise"
Y_text = "paragraph"
X = n_gram(X_text, 2)
Y = n_gram(Y_text, 2)

print(f"和集合: {set(X) | set(Y)}")
print(f"積集合: {set(X) & set(Y)}")
print(f"差集合: {set(X) - set(Y)}")
print("se" in (set(X) & set(Y)))
