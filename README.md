# Week 3: Statistical Analysis and Hypothesis Testing (Restaurant Tips)

Internship: **Virtual Data Science with Python Apprentice Intern** at YuvaIntern
Intern: **Yugank Charnotia**

## Question
What influences how generously restaurant customers tip (tip as a percentage of the bill)?

## Dataset
`tips` dataset (244 bills, 7 columns) from the [Seaborn data repository](https://github.com/mwaskom/seaborn-data). No missing values.

## Hypotheses and tests
| ID | Hypothesis (H0: no difference / no relationship) | Test | Result |
|---|---|---|---|
| H1 | Tip % same at lunch and dinner | Welch t-test | p = 0.514, not significant |
| H2 | Tip % same for smokers and non-smokers | Welch t-test | p = 0.682, not significant |
| H3 | Tip % same for female and male payers | Welch t-test | p = 0.254, not significant |
| H4 | Tip % same across Thu/Fri/Sat/Sun | One-way ANOVA + Tukey | p = 0.469, not significant |
| H5 | No relationship between bill size and tip % | Pearson correlation | r = -0.34, p < 0.001, significant |
| H6 | Party size independent of weekday/weekend | Chi-square | p = 0.0002, V = 0.24, significant |
| H7 | Regression of tip $ on all factors | OLS with HC3 robust errors | Only total bill is a clear predictor |

## Methods used
- Assumption checks: Shapiro-Wilk, Levene, Q-Q plots, Breusch-Pagan
- Robust backups: Mann-Whitney, Kruskal-Wallis, Fisher's exact, bootstrap CIs, HC3 standard errors
- Effect sizes: Cohen's d, eta-squared, Cramer's V; power analysis
- Holm correction for multiple testing; sensitivity analysis without outliers

## Key conclusions
- Larger bills get a smaller tip percentage (about 2.3 points less per extra $10), though bigger dollar tips.
- Large parties are about 3x more likely on weekends.
- No significant evidence of differences by meal time, smoking status, sex of payer or day. These null results are inconclusive for small effects (the study could detect about 2.5 points or more).

## Folder structure
```
analysis.py          all tests and figures
results_output.txt   printed output of the script
requirements.txt
data/tips.csv
figures/             7 charts
```

## How to run
```bash
pip install -r requirements.txt
python analysis.py
```

## Charts
![fig1](figures/fig1_distribution_assumptions.png)
![fig2](figures/fig2_group_comparisons.png)
![fig3](figures/fig3_anova_day.png)
![fig4](figures/fig4_correlation.png)
![fig5](figures/fig5_chisquare.png)
![fig6](figures/fig6_pvalues.png)
![fig7](figures/fig7_regression_diagnostics.png)

## Tools
Python, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn
