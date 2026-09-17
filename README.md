# Purchase Prediction: Regularized Logistic Regression

Predicting whether an online shopping session ends in a purchase, using logistic regression with LASSO and ridge regularization in R.

**Report:** [shopping_analysis.ipynb](shopping_analysis.ipynb) (R kernel; renders on GitHub) · [HTML export](shopping_analysis.html)

## Problem

Most e-commerce sessions end without a purchase. Given a session's browsing behaviour — pages visited, time on each page type, bounce/exit rates, month, visitor type — can we predict whether it converts?

Data: [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI), 12,330 sessions × 18 variables, one session per user over one year.

## Approach

1. **EDA** — class balance (≈15% of sessions convert, so accuracy is a misleading metric), summary statistics, correlation plot of numeric features.
2. **Multicollinearity** — dropped features with pairwise correlation > 0.6 (`BounceRates`, `ProductRelated`, `Informational`, `Administrative`) and a categorical feature with aliased levels (`Browser`); verified with VIF (all < 5).
3. **Models** — plain logistic regression baseline, then LASSO and ridge logistic regression tuned with 10-fold cross-validation, optimizing AUC (`glmnet`).
4. **Threshold** — used a 0.3 probability cutoff instead of 0.5 to recover more of the minority (purchase) class.
5. **Evaluation** — 70/30 stratified split; test set untouched until the final assessment.

## Results

| Model | CV AUC | Features kept |
| --- | --- | --- |
| **LASSO (final)** | **0.911** | 3 of 18 |
| Ridge | 0.893 | all 18 |

Final LASSO model on the held-out test set: **AUC 0.909**, accuracy 88.7%, sensitivity 0.377, balanced accuracy 0.677.

L1 regularization cut 18 predictors down to three: `PageValues` (strongly positive), `ExitRates` (negative), and `Month = November`. The model ranks sessions well (high AUC) but still misses many true purchases, which is the expected consequence of the class imbalance — discussed at the end of the notebook along with next steps (resampling, tree-based models).

## Stack

R · `glmnet` · `caret` · `pROC` · `car` · `rsample` · `dplyr` · `ggplot2` · `corrplot`

## Running it

Open `shopping_analysis.ipynb` in Jupyter with an R kernel (IRkernel) and run all cells. The dataset is loaded from a URL, so no download is needed. Packages: `install.packages(c("glmnet","caret","pROC","car","rsample","dplyr","ggplot2","ggpubr","corrplot","broom"))`.
