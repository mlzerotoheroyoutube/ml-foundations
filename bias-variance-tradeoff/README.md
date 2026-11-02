# Bias-Variance Tradeoff

**Level:** Intermediate  ·  **Topic:** ML Foundations

▶️ **Watch the video:** https://youtu.be/2q1OGMLJvq4

> Under squared loss, expected test error splits exactly into **bias²** (systematic error from too-rigid assumptions) + **variance** (sensitivity to the particular training sample) + **irreducible noise**. Classically, more flexible models trade lower bias for higher variance, giving a U-shaped test-error curve. Modern overparameterised models break the U: past the interpolation threshold, variance can fall again (double descent).

## The intuition
Picture four archers, each shooting 10 arrows at a target.

- **Archer A** groups all arrows tightly, but 20 cm to the left of the bullseye. Consistent but consistently wrong: **high bias, low variance**.
- **Archer B** has arrows centred on the bullseye *on average*, but scattered all over the target. Right on average, wildly inconsistent: **low bias, high variance**.
- **Archer C** is tight *and* centred, the dream.
- **Archer D** is scattered *and* off-centre, the nightmare.

Now swap archers for models and "each shot" for "train the model on a different random sample of data". A model that's too simple, like drawing a straight line through data that curves, makes the same mistake no matter which data you give it. That's bias. A model that's too flexible, like a squiggly curve that passes through every single point, changes wildly whenever the data changes slightly. That's variance.

Example with numbers: predicting a student's test score. A model that always says "70" has huge bias; it ignores everything. A model that memorises "the student in seat 14 got 93" has huge variance; it's useless for a new class. The sweet spot is in between: "score goes up about 5 points per hour of study".

There's also a third piece you can never remove: **noise**. Some students just have a bad day. No model can predict that.

## Core notes
- **Bias** = error from wrong assumptions; average prediction ≠ truth. Symptom: high training *and* validation error (underfitting).
- **Variance** = error from sensitivity to the training sample. Symptom: low training error, much higher validation error (overfitting).
- **Noise** $\sigma^2$ = irreducible (Bayes) error.
- Classical knobs: model complexity, Regularization strength, number of features, training-set size (more data → lower variance).
- **Reduce bias:** bigger or more flexible model, better features, less regularisation, Boosting.
- **Reduce variance:** more data, regularisation, Dropout, Data Augmentation, Bagging / Random Forests, Early Stopping.
- The decomposition is exact only for squared loss; for classification it is heuristic.
- **Double Descent:** beyond the interpolation threshold, test error can fall again. "Bigger is better" in modern deep learning is consistent with this.
- Diagnose with **learning curves** (error vs $n$) and **validation curves** (error vs complexity).

## The math
**Setup.** $y=f(x)+\varepsilon$, $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}(\varepsilon)=\sigma^2$, independent of training set $S$. Let $\bar f(x)=\mathbb{E}_S[\hat f_S(x)]$.

**Decomposition.** At a fixed test point $x$:
$$
\begin{aligned}
\mathbb{E}_{S,\varepsilon}\big[(y-\hat f_S(x))^2\big]
&=\mathbb{E}\big[(f(x)+\varepsilon-\hat f_S(x))^2\big]\\
&=\sigma^2+\mathbb{E}_S\big[(f(x)-\hat f_S(x))^2\big]\\
&=\underbrace{\sigma^2}_{\text{noise}}+\underbrace{\big(f(x)-\bar f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}_S\big[(\hat f_S(x)-\bar f(x))^2\big]}_{\text{variance}}.
\end{aligned}
$$
The cross terms vanish because $\mathbb{E}\varepsilon=0$ and $\mathbb{E}_S[\hat f_S(x)-\bar f(x)]=0$.

**Example: k-NN regression.** With fixed design and neighbours at distance $\approx 0$, $\operatorname{Var}=\sigma^2/k$, while bias grows with the neighbourhood radius. Small $k$ → low bias, high variance.

**Example: ridge regression.** $\hat w_\lambda=(X^\top X+\lambda I)^{-1}X^\top y$. In the eigenbasis of $X^\top X$ (eigenvalues $d_j$), each coordinate is shrunk by $d_j/(d_j+\lambda)$: bias grows with $\lambda$, variance $\sigma^2\sum_j d_j/(d_j+\lambda)^2$ falls with $\lambda$.

**Bagging.** Averaging $B$ predictors each with variance $\sigma_h^2$ and pairwise correlation $\rho$:
$$
\operatorname{Var}\Big(\tfrac1B\textstyle\sum_b \hat f_b\Big)=\rho\,\sigma_h^2+\frac{1-\rho}{B}\,\sigma_h^2 .
$$

## Run the code
```bash
pip install numpy scikit-learn
python bias_variance_tradeoff.py
```

## Check yourself
<details><summary>Derive the bias-variance decomposition for squared loss.</summary>

Write $y=f(x)+\varepsilon$ and let $\bar f(x)=\mathbb{E}_S\hat f_S(x)$. Expand $\mathbb{E}[(y-\hat f)^2]=\mathbb{E}[\varepsilon^2]+\mathbb{E}[(f-\hat f)^2]$ since $\varepsilon$ is zero-mean and independent of $\hat f$. Then add and subtract $\bar f$: $\mathbb{E}[(f-\bar f+\bar f-\hat f)^2]=(f-\bar f)^2+\mathbb{E}[(\hat f-\bar f)^2]$, the cross-term vanishing because $\mathbb{E}[\hat f-\bar f]=0$. The result is noise + bias² + variance.
</details>

<details><summary>Your model has 2% training error and 15% validation error. Human error is 1%. What do you do?</summary>

Avoidable bias is about 1 point (2% vs 1%), while variance is about 13 points, so variance dominates. Get more data or use augmentation, add regularisation (weight decay, dropout), use early stopping, simplify the model or ensemble. Also check that validation and training come from the same distribution: a large gap can also be distribution mismatch rather than variance, which you can diagnose with a train-dev set drawn from the training distribution.
</details>

<details><summary>How do bagging and boosting relate to bias and variance?</summary>

Bagging trains low-bias, high-variance learners (deep trees) on bootstrap samples and averages them; averaging cuts variance by roughly $(1-\rho)/B$ while leaving bias unchanged, and random forests decorrelate trees further via feature subsampling. Boosting fits weak, high-bias learners sequentially to residuals or reweighted errors, primarily reducing bias; it can increase variance and overfit without shrinkage, subsampling and early stopping.
</details>

<details><summary>Does double descent contradict the bias-variance tradeoff?</summary>

It contradicts the U-shaped *tradeoff* but not the *decomposition*, which remains an identity for squared loss. In the overparameterised regime, the optimiser selects a particular interpolating solution (e.g. minimum norm), whose variance decreases as more parameters let it spread the fit over many directions. The variance peak sits at the interpolation threshold, where the system is barely able to fit the data and the solution norm blows up. So parameter count is the wrong complexity measure; the norm or effective dimension of the selected solution is closer to the right one.
</details>

<details><summary>Why does more data reduce variance but not bias?</summary>

Variance measures how much the fitted model changes across training samples; as $n$ grows, estimates concentrate (often at rate $1/n$ for parametric models), so variance shrinks. Bias is the gap between the best average prediction of the model family and the truth; if the family can't represent the target (a line for a sine wave), no amount of data closes that gap. That is why learning curves for high-bias models plateau above the noise floor.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.