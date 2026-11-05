# Runs the whole R analysis. Run from the repository root:  Rscript R/run_all.R
# 1) models 1-14 from master_ml_benchmark.R (stops before its caret XGBoost block,
#    which fails with xgboost >= 3.x), 2) XGBoost via the xgboost API, 3) ROC/AUC,
#    confusion matrices and the 15-model leaderboard, 4) EDA and statistical tests.
dir.create("figs", showWarnings = FALSE)
lines <- readLines("R/master_ml_benchmark.R")
cut <- grep("MODEL: XGBOOST", lines)[1] - 2
eval(parse(text = lines[seq_len(cut)]), envir = globalenv())
source("R/xgboost_direct.R")
source("R/extra_results.R")
source("R/eda_and_statistics.R")
message("Done. Figures are in figs/, statistics in stats_output.txt")
