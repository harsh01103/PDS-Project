# ==============================================================================
# EDA AND STATISTICAL ANALYSIS - Study Stock options project
# Course : Data Analytics Solutions using R, VIT Chennai
# Input  : data/options_trade_signals_sample.csv  (modelling dataset, 500 rows)
#          data/preprocessed_features_sample.csv  (option chain features)
#          data/raw_sample_dataset.csv            (raw WebSocket ticks)
# Output : figs/*.png and stats_output.txt
# ==============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(ggplot2); library(scales)
})

dir.create("figs", showWarnings = FALSE)
theme_rep <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom",
        panel.grid.minor = element_blank())
col_ce <- "#1f77b4"; col_pe <- "#d62728"; col_p <- "#2a9d8f"; col_u <- "#e76f51"

# ---- 1. Load and build the economic target (same rule as the benchmark script) ----
df_raw <- read_csv("data/options_trade_signals_sample.csv", show_col_types = FALSE) %>% arrange(timestamp)
friction_drag <- 0.35 * df_raw$spread + 0.15
gross_alpha   <- 0.08 * df_raw$z_vwoim + 0.06 * df_raw$z_oi_flow + 0.20 * (df_raw$confidence - 0.90)
df <- df_raw %>%
  mutate(net_return = gross_alpha - friction_drag,
         target = factor(ifelse(net_return > 0, "Profitable", "Unprofitable"),
                         levels = c("Profitable", "Unprofitable")),
         log_volume = log1p(volume))
feat <- c("confidence", "spread", "log_volume", "z_vwoim", "z_oi_flow")

sink("stats_output.txt", split = TRUE)
cat("=== DATASET OVERVIEW ===\n")
cat("Rows:", nrow(df), " Columns:", ncol(df_raw), "\n")
cat("Underlyings:\n"); print(table(df$underlying))
cat("Option type:\n"); print(table(df$option_type))
cat("Expiries:\n"); print(table(df$expiry))
cat("Missing values per column (modelling dataset):\n"); print(colSums(is.na(df_raw)))
cat("\nClass distribution of target:\n"); print(table(df$target)); print(round(prop.table(table(df$target)), 4))

# ---- 2. Descriptive statistics ----
cat("\n=== DESCRIPTIVE STATISTICS (feature columns) ===\n")
desc <- df %>% select(all_of(feat)) %>%
  pivot_longer(everything(), names_to = "feature") %>%
  group_by(feature) %>%
  summarise(mean = mean(value), sd = sd(value), median = median(value),
            min = min(value), max = max(value), skew = mean((value - mean(value))^3) / sd(value)^3,
            .groups = "drop")
print(as.data.frame(desc), digits = 4)

# ---- 3. Hypothesis tests: features by target class ----
cat("\n=== WELCH T-TEST AND WILCOXON TEST: feature vs target class ===\n")
tests <- lapply(feat, function(f) {
  x <- df[[f]][df$target == "Profitable"]; y <- df[[f]][df$target == "Unprofitable"]
  tt <- t.test(x, y); wt <- suppressWarnings(wilcox.test(x, y))
  data.frame(feature = f, mean_profitable = mean(x), mean_unprofitable = mean(y),
             t = unname(tt$statistic), t_p = tt$p.value, wilcoxon_p = wt$p.value)
})
tests <- do.call(rbind, tests); print(tests, digits = 4, row.names = FALSE)

# ---- 4. Association tests ----
cat("\n=== CHI-SQUARE: option_type vs target ===\n")
print(chisq.test(table(df$option_type, df$target)))
cat("\n=== CHI-SQUARE: underlying vs target ===\n")
print(chisq.test(table(df$underlying, df$target)))
cat("\n=== ONE-WAY ANOVA: spread by underlying ===\n")
print(summary(aov(spread ~ underlying, data = df)))
cat("\n=== SHAPIRO-WILK NORMALITY (feature columns) ===\n")
for (f in feat) { s <- shapiro.test(df[[f]]); cat(sprintf("%-12s W = %.4f  p = %.3g\n", f, s$statistic, s$p.value)) }

# ---- 5. Correlation ----
cat("\n=== PEARSON CORRELATION MATRIX ===\n")
cm <- cor(df %>% select(all_of(feat), net_return))
print(round(cm, 3))
cat("\nSelected correlation tests:\n")
for (pr in list(c("spread", "net_return"), c("z_vwoim", "net_return"), c("z_oi_flow", "net_return"), c("confidence", "net_return"))) {
  ct <- cor.test(df[[pr[1]]], df[[pr[2]]])
  cat(sprintf("%-10s vs %-10s r = %6.3f  p = %.3g\n", pr[1], pr[2], ct$estimate, ct$p.value))
}
sink()

# ---- 6. Figures ----
# Fig 1: class balance
p1 <- ggplot(as.data.frame(table(df$target)), aes(Var1, Freq, fill = Var1)) +
  geom_col(width = 0.6) + geom_text(aes(label = Freq), vjust = -0.4, fontface = "bold") +
  scale_fill_manual(values = c(Profitable = col_p, Unprofitable = col_u), guide = "none") +
  labs(title = "Target class balance (net of friction)", x = NULL, y = "Signals") + theme_rep
ggsave("figs/eda_1_class_balance.png", p1, width = 7, height = 4.5, dpi = 200)

# Fig 2: feature histograms
long <- df %>% select(target, all_of(feat)) %>% pivot_longer(-target, names_to = "feature")
p2 <- ggplot(long, aes(value, fill = target)) +
  geom_histogram(bins = 30, alpha = 0.75, position = "identity") +
  facet_wrap(~feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c(Profitable = col_p, Unprofitable = col_u), name = NULL) +
  labs(title = "Feature distributions by target class", x = NULL, y = "Count") + theme_rep
ggsave("figs/eda_2_feature_histograms.png", p2, width = 11, height = 6.5, dpi = 200)

# Fig 3: boxplots
p3 <- ggplot(long, aes(target, value, fill = target)) + geom_boxplot(outlier.size = 1, alpha = 0.8) +
  facet_wrap(~feature, scales = "free_y", ncol = 5) +
  scale_fill_manual(values = c(Profitable = col_p, Unprofitable = col_u), guide = "none") +
  labs(title = "Feature spread by target class", x = NULL, y = NULL) + theme_rep +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
ggsave("figs/eda_3_boxplots.png", p3, width = 12, height = 4.8, dpi = 200)

# Fig 4: correlation heatmap
cl <- as.data.frame(as.table(cm)); names(cl) <- c("a", "b", "r")
p4 <- ggplot(cl, aes(a, b, fill = r)) + geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", r)), size = 4) +
  scale_fill_gradient2(low = col_pe, mid = "white", high = col_ce, limits = c(-1, 1), name = "r") +
  labs(title = "Correlation matrix: features and net return", x = NULL, y = NULL) + theme_rep +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("figs/eda_4_correlation.png", p4, width = 8, height = 6, dpi = 200)

# Fig 5 and 6: option chain features (recorded NIFTY / BANKNIFTY)
pf <- read_csv("data/preprocessed_features_sample.csv", show_col_types = FALSE)
nifty <- pf %>% filter(underlying == "NIFTY") %>% group_by(strike, option_type) %>%
  summarise(oi = max(oi, na.rm = TRUE), .groups = "drop")
p5 <- ggplot(nifty, aes(factor(strike), oi, fill = option_type)) +
  geom_col(position = "dodge", width = 0.75) +
  scale_fill_manual(values = c(CE = col_ce, PE = col_pe), name = "Option type") +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(title = "NIFTY open interest by strike (recorded session, 18 Jun 2026)", x = "Strike", y = "Open interest") +
  theme_rep + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("figs/eda_5_open_interest.png", p5, width = 10, height = 5, dpi = 200)

pm <- pf %>% filter(spot > 0, bid > 0, ask > 0, underlying == "NIFTY") %>%
  mutate(moneyness = strike / spot, spread = ask - bid)
p6 <- ggplot(pm, aes(moneyness, spread, color = option_type)) + geom_point(alpha = 0.55, size = 1.8) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  scale_color_manual(values = c(CE = col_ce, PE = col_pe), name = "Option type") +
  labs(title = "Bid-ask spread vs moneyness (NIFTY, recorded ticks)", x = "Strike / spot", y = "Spread (INR)") + theme_rep
ggsave("figs/eda_6_spread_moneyness.png", p6, width = 10, height = 5, dpi = 200)

# Fig 7: feed latency
rw <- read_csv("data/raw_sample_dataset.csv", show_col_types = FALSE)
lat <- rw %>% filter(!is.na(latency_ms), abs(latency_ms) < 5000)
p7 <- ggplot(lat, aes(latency_ms)) + geom_histogram(bins = 40, fill = col_ce, color = "white") +
  geom_vline(xintercept = median(lat$latency_ms), color = col_u, linewidth = 1) +
  labs(title = "WebSocket feed latency (receive minus exchange time)",
       subtitle = sprintf("Median %.0f ms across %d raw ticks", median(lat$latency_ms), nrow(lat)),
       x = "Latency (ms)", y = "Ticks") + theme_rep
ggsave("figs/eda_7_latency.png", p7, width = 8, height = 4.8, dpi = 200)

cat("Raw ticks:", nrow(rw), " Preprocessed rows:", nrow(pf), " Signals:", nrow(df_raw), "\n")
message("EDA figures and stats_output.txt written")
