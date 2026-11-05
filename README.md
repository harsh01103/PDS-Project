# Options Trading Data Science Project (Study Stock) - R analysis

Course project for **Data Analytics Solutions using R**, Vellore Institute of Technology (Chennai Campus).

- **Team:** Harsh Vardhan Ajmera (24BDS1060), Ahana Manna (24BDS1102)
- **Faculty:** Dr. Vijayalakshmi A
- **Main project repository (web app and data engine):** https://github.com/harsh01103/stockezee

## What this is
We recorded live NIFTY and BANKNIFTY option ticks from the NSE, built a 500-signal dataset and compared 15 classifiers in R on a time-ordered test set of 94 signals. Accuracy ranges from 89.4% (CART) to 100% (probit regression, linear SVM).

**Please read this caveat:** the "Profitable" label is computed from the same five input features, so the high scores show that the pipeline works, not that the market is predictable. Our drafted research paper (not published) repeats the test with real future-price labels and finds an AUC of at most 0.60 and no profitable filter after costs.

## Folder layout
- `R/` - the four R scripts and `run_all.R`
- `data/` - the four CSV samples exported from our recorded data
- `outputs/` - R console output, 15-model leaderboard, AUC table, statistics output
- `figs/` - EDA plots, result plots (accuracy, ROC, confusion matrices, XGBoost importance), pipeline diagram
- `screenshots/` - Study Stock web app screens and the first page of the draft paper
- `report/` - project report (PDF and Word)

## How to run
Install the packages once:
```r
install.packages(c("MASS","class","nnet","e1071","rpart","rpart.plot","randomForest","ipred",
                   "dplyr","readr","tidyr","caret","ggplot2","scales","pROC","xgboost"))
```
Then, from the repository root: `Rscript R/run_all.R` (or `source("R/run_all.R")` in R).

## Scripts
- `R/master_ml_benchmark.R` - 14 models (logit, probit, LDA, QDA, KNN k=5/11, Naive Bayes, three SVMs, CART, bagging, random forest, neural net), chronological 80/20 split with a 2% embargo. Its last block (caret XGBoost) fails with xgboost 3.x, so `run_all.R` stops before it.
- `R/xgboost_direct.R` - the same XGBoost grid and 5-fold CV run directly through the xgboost package.
- `R/extra_results.R` - ROC curves and AUC, confusion matrices, XGBoost importance, final 15-model leaderboard.
- `R/eda_and_statistics.R` - EDA figures, Welch/Wilcoxon, chi-square, ANOVA, Shapiro-Wilk and correlation tests.

## Data
Samples exported from per-minute Parquet files recorded through the Upstox WebSocket feed. The full recording (57,740 files in 48 session folders, 18 Jun to 20 Oct 2026) stays on the team's PC and is not in this repository.

## Research paper status
The paper is a **draft and has not been published or accepted anywhere**. Only its first page is shown in `screenshots/`.
