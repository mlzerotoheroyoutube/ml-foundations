"""Bias-Variance Tradeoff: code from the ML Zero to Hero explainer.

Video: https://youtu.be/2q1OGMLJvq4
"""

import numpy as np

# Monte Carlo bias-variance decomposition for polynomial regression of increasing degree
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
sigma, n, trials = 0.3, 30, 500
x_test = np.linspace(0.1, 0.9, 50)

for degree in [1, 3, 10]:   # underfit, about right, overfit
    preds = np.empty((trials, x_test.size))
    for t in range(trials):
        x = rng.uniform(0, 1, n)
        y = f(x) + sigma * rng.normal(size=n)
        coefs = np.polyfit(x, y, degree)
        preds[t] = np.polyval(coefs, x_test)
    bias2 = np.mean((preds.mean(0) - f(x_test)) ** 2)
    var = np.mean(preds.var(0))
    print(f"degree {degree}: bias²={bias2:.3f}  variance={var:.3f}  "
          f"expected test MSE≈{bias2 + var + sigma**2:.3f}")

# --------------------------------------------------------------------------

# Bagging reduces variance: one deep tree vs a bagged ensemble, over repeated training sets
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0, 1, 100)[:, None]
for name, make in [("single tree", lambda s: DecisionTreeRegressor(random_state=s)),
                   ("bagged trees", lambda s: BaggingRegressor(DecisionTreeRegressor(), n_estimators=50, random_state=s))]:
    P = []
    for s in range(100):
        x = rng.uniform(0, 1, (60, 1)); y = f(x[:, 0]) + 0.3 * rng.normal(size=60)
        P.append(make(s).fit(x, y).predict(x_test))
    P = np.array(P)
    print(f"{name:12s} bias²={np.mean((P.mean(0) - f(x_test[:, 0]))**2):.3f}  variance={P.var(0).mean():.3f}")
