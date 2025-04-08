import joblib

clf = joblib.load("ch07/model.joblib")
vocabulary_ = joblib.load("ch07/vocabulary_.joblib")
coefs = clf.coef_

for c in coefs:
    d = dict(zip(vocabulary_, c))
    d_top = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    print(d_top)
    d_bottom = sorted(d.items(), key=lambda x: abs(x[1]), reverse=False)[:10]
    print(d_bottom)
