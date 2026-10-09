import os, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, RED, GREEN = "#2E86AB", "#F3A712", "#9AA0A6", "#E4572E", "#59A96A"
ALPHA = 0.05

# ------------------------------------------------------------ DATA
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
try:
    d = pd.read_csv(URL); d.to_csv("data/tips.csv", index=False)
except Exception:
    d = pd.read_csv("data/tips.csv")
d["tip_pct"] = d.tip / d.total_bill * 100
d["daytype"] = np.where(d.day.isin(["Sat", "Sun"]), "Weekend", "Weekday")
d["party"] = np.where(d["size"] >= 3, "Large (3+)", "Small (1-2)")
print("rows", len(d), "missing", d.isnull().sum().sum(), "duplicates", d.duplicated().sum())
print(d[["total_bill", "tip", "size", "tip_pct"]].describe().round(2))
q1, q3 = d.tip_pct.quantile([.25, .75]); UB = q3 + 1.5 * (q3 - q1)
out = d[d.tip_pct > UB]; print("IQR upper bound", round(UB, 2), "n outliers", len(out), "max", round(d.tip_pct.max(), 1))
print("skew", round(stats.skew(d.tip_pct), 2), "kurtosis", round(stats.kurtosis(d.tip_pct), 2))
print("Shapiro tip_pct", stats.shapiro(d.tip_pct))

# ------------------------------------------------------------ HELPERS
def boot_diff(x, y, n=10000):
    b = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])
def boot_mean_ci(x, n=5000):
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5])
def two_group(df, col, a, b):
    x = df[df[col] == a].tip_pct.values; y = df[df[col] == b].tip_pct.values
    t = stats.ttest_ind(x, y, equal_var=False); u = stats.mannwhitneyu(x, y)
    se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y)); diff = x.mean() - y.mean()
    ci = diff + np.array([-1, 1]) * stats.t.ppf(.975, t.df) * se
    sp = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / (len(x)+len(y)-2))
    return dict(n1=len(x), m1=x.mean(), s1=x.std(ddof=1), n2=len(y), m2=y.mean(), s2=y.std(ddof=1),
                diff=diff, t=t.statistic, df=t.df, p=t.pvalue, ci=ci, bci=boot_diff(x, y), d=diff/sp,
                mwu=u.pvalue, lev=stats.levene(x, y).pvalue,
                sh1=stats.shapiro(x).pvalue, sh2=stats.shapiro(y).pvalue)
def show(name, r):
    print(f"\n{name}: n={r['n1']}/{r['n2']} means {r['m1']:.2f}/{r['m2']:.2f} sd {r['s1']:.2f}/{r['s2']:.2f}")
    print(f"  diff={r['diff']:.2f} Welch t={r['t']:.3f} df={r['df']:.1f} p={r['p']:.4f} CI95=[{r['ci'][0]:.2f},{r['ci'][1]:.2f}] bootCI=[{r['bci'][0]:.2f},{r['bci'][1]:.2f}] d={r['d']:.3f}")
    print(f"  MWU p={r['mwu']:.4f} Levene p={r['lev']:.4f} Shapiro p={r['sh1']:.4g}/{r['sh2']:.4g}")

# ------------------------------------------------------------ H1-H3 two-group tests
H1 = two_group(d, "time", "Lunch", "Dinner"); show("H1 Lunch vs Dinner", H1)
H2 = two_group(d, "smoker", "Yes", "No"); show("H2 Smoker vs Non-smoker", H2)
H3 = two_group(d, "sex", "Female", "Male"); show("H3 Female vs Male payer", H3)
pw = TTestIndPower()
mde = pw.solve_power(nobs1=H1["n1"], ratio=H1["n2"]/H1["n1"], alpha=ALPHA, power=.8)
print("\nPower: min detectable d (lunch/dinner) =", round(float(mde), 3), "=> pct points", round(float(mde*d.tip_pct.std()), 2))
print("Achieved power for observed d:", round(float(pw.power(effect_size=abs(H1['d']), nobs1=H1['n1'], ratio=H1['n2']/H1['n1'], alpha=ALPHA)), 3))

# ------------------------------------------------------------ H4 ANOVA by day
order = ["Thur", "Fri", "Sat", "Sun"]
groups = [d[d.day == k].tip_pct.values for k in order]
print("\nH4 by day:"); print(d.groupby("day").tip_pct.agg(["count", "mean", "std"]).loc[order].round(2))
F = stats.f_oneway(*groups); KW = stats.kruskal(*groups); LV = stats.levene(*groups)
aov = sm.stats.anova_lm(smf.ols("tip_pct ~ C(day)", d).fit(), typ=2)
eta2 = aov.sum_sq.iloc[0] / aov.sum_sq.sum()
print(f"ANOVA F(3,240)={F.statistic:.3f} p={F.pvalue:.4f} eta2={eta2:.4f} | Kruskal H={KW.statistic:.3f} p={KW.pvalue:.4f} | Levene p={LV.pvalue:.3f}")
print(pairwise_tukeyhsd(d.tip_pct, d.day))

# ------------------------------------------------------------ H5 correlations
r_amt = stats.pearsonr(d.total_bill, d.tip); r_pct = stats.pearsonr(d.total_bill, d.tip_pct)
s_pct = stats.spearmanr(d.total_bill, d.tip_pct)
print(f"\nH5a bill vs tip $: r={r_amt.statistic:.3f} p={r_amt.pvalue:.3g} CI={np.round(r_amt.confidence_interval(),3)}")
print(f"H5b bill vs tip%: pearson r={r_pct.statistic:.3f} p={r_pct.pvalue:.3g} CI={np.round(r_pct.confidence_interval(),3)} | spearman rho={s_pct.statistic:.3f} p={s_pct.pvalue:.3g}")
simple = smf.ols("tip_pct ~ total_bill", d).fit(); print(simple.params.round(3).to_dict(), simple.conf_int().round(3).values.tolist(), "R2", round(simple.rsquared, 3))
amt = smf.ols("tip ~ total_bill", d).fit(); print("tip$ ~ bill:", amt.params.round(3).to_dict(), amt.conf_int().round(3).values.tolist(), "R2", round(amt.rsquared, 3))

# ------------------------------------------------------------ H6 chi-square
ct = pd.crosstab(d.daytype, d.party); chi = stats.chi2_contingency(ct, correction=False)
V = np.sqrt(chi.statistic / ct.values.sum())
print("\nH6 chi-square:"); print(ct); print("expected\n", np.round(chi.expected_freq, 1))
print(f"chi2({chi.dof})={chi.statistic:.3f} p={chi.pvalue:.4f} Cramer's V={V:.3f}")
share = ct.div(ct.sum(axis=1), axis=0)*100; print(share.round(1))
resid = (ct.values - chi.expected_freq) / np.sqrt(chi.expected_freq); print("std residuals\n", np.round(resid, 2))
fish = stats.fisher_exact(ct.values); print("Fisher exact p", round(fish.pvalue, 4), "odds ratio", round(fish.statistic, 3))

# ------------------------------------------------------------ H7 multiple regression
mod = smf.ols("tip ~ total_bill + size + C(sex) + C(smoker) + C(time) + C(day)", d).fit()
modr = mod.get_robustcov_results(cov_type="HC3")
print("\nH7 OLS:\n", mod.summary().tables[1]); print("R2", round(mod.rsquared, 3), "adjR2", round(mod.rsquared_adj, 3))
print("HC3 robust p-values:", dict(zip(mod.params.index, np.round(modr.pvalues, 4))))
print("Breusch-Pagan p:", round(sm.stats.diagnostic.het_breuschpagan(mod.resid, mod.model.exog)[1], 5))
print("Residual Shapiro p:", stats.shapiro(mod.resid).pvalue)

# ------------------------------------------------------------ MULTIPLE TESTING (family of six hypotheses)
names = ["H1 Lunch vs Dinner", "H2 Smoker vs Non-smoker", "H3 Female vs Male", "H4 Day of week (ANOVA)", "H5 Bill vs tip %", "H6 Weekend vs party size"]
raw = np.array([H1["p"], H2["p"], H3["p"], F.pvalue, r_pct.pvalue, chi.pvalue])
holm = multipletests(raw, method="holm")[1]
print("\nSummary (raw / Holm):")
for n, a, b in zip(names, raw, holm): print(f"  {n:28s} raw={a:.4g} holm={b:.4g} -> {'REJECT H0' if b < ALPHA else 'fail to reject'}")

# ------------------------------------------------------------ SENSITIVITY: remove IQR outliers
dn = d[d.tip_pct <= UB]
s1 = two_group(dn, "time", "Lunch", "Dinner"); s2 = two_group(dn, "smoker", "Yes", "No"); s3 = two_group(dn, "sex", "Female", "Male")
sF = stats.f_oneway(*[dn[dn.day == k].tip_pct.values for k in order]); sr = stats.pearsonr(dn.total_bill, dn.tip_pct)
print("\nSensitivity (outliers removed, n=%d): H1 p=%.4f H2 p=%.4f H3 p=%.4f H4 p=%.4f H5 r=%.3f p=%.3g" % (len(dn), s1["p"], s2["p"], s3["p"], sF.pvalue, sr.statistic, sr.pvalue))
print("  group means no-outlier time:", dn.groupby("time").tip_pct.mean().round(2).to_dict(), "sd", dn.groupby("time").tip_pct.std().round(2).to_dict())
print("  H1 diff", round(s1['diff'], 2), "CI", np.round(s1['ci'], 2), " H2 diff", round(s2['diff'], 2), np.round(s2['ci'], 2), " H3", round(s3['diff'], 2), np.round(s3['ci'], 2))
print("  Levene smoker no-outlier", round(s2['lev'], 3))

# ================================================================ FIGURES
# Fig 1: distribution + QQ
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
sns.histplot(d.tip_pct, bins=40, kde=True, color=BLUE, ax=ax[0])
ax[0].axvline(d.tip_pct.mean(), color=RED, ls="--", label=f"Mean {d.tip_pct.mean():.1f}%")
ax[0].axvline(d.tip_pct.median(), color=GREEN, ls="--", label=f"Median {d.tip_pct.median():.1f}%")
ax[0].annotate("Extreme tip:\n71% of a $7.25 bill", xy=(71, 1), xytext=(45, 18), arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)
ax[0].set_xlabel("Tip as % of bill"); ax[0].set_title("Tip % is right-skewed with a long tail", loc="left", fontweight="bold"); ax[0].legend(frameon=False)
(osm, osr), (slope, icpt, _) = stats.probplot(d.tip_pct, dist="norm")
ax[1].scatter(osm, osr, s=14, color=BLUE, alpha=.7); ax[1].plot(osm, slope*np.array(osm)+icpt, color=RED)
ax[1].set_xlabel("Theoretical quantiles"); ax[1].set_ylabel("Observed tip %"); ax[1].set_title("Q-Q plot: tails depart from normal", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig1_distribution_assumptions.png", dpi=160); plt.close()

# Fig 2: group comparisons
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
for ax, (col, lev, res, ttl) in zip(axs, [("time", ["Lunch", "Dinner"], H1, "H1: Lunch vs Dinner"), ("smoker", ["Yes", "No"], H2, "H2: Smoker vs Non-smoker"), ("sex", ["Female", "Male"], H3, "H3: Female vs Male payer")]):
    sns.stripplot(data=d, x=col, y="tip_pct", order=lev, color=GREY, alpha=.45, size=3.5, jitter=.25, ax=ax)
    for i, l in enumerate(lev):
        x = d[d[col] == l].tip_pct.values; ci = boot_mean_ci(x)
        ax.errorbar(i, x.mean(), yerr=[[x.mean()-ci[0]], [ci[1]-x.mean()]], fmt="o", color=BLUE if i == 0 else ORANGE, ms=9, capsize=6, lw=2.5, zorder=5)
    ax.set_title(ttl, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Tip % of bill" if ax is axs[0] else "")
    ax.text(.5, .97, f"Welch p = {res['p']:.2f}\ndiff = {res['diff']:+.2f} pts", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.set_ylim(0, 45)
fig.suptitle("Group means with bootstrap 95% CIs: the intervals overlap heavily (y-axis cut at 45%)", x=.01, ha="left", fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/fig2_group_comparisons.png", dpi=160); plt.close()

# Fig 3: ANOVA by day
fig, ax = plt.subplots(figsize=(8.5, 4.6))
sns.boxplot(data=d, x="day", y="tip_pct", order=order, color="#DCE9F2", fliersize=0, ax=ax, width=.55)
sns.stripplot(data=d, x="day", y="tip_pct", order=order, color=BLUE, alpha=.5, size=3.5, ax=ax)
ms = d.groupby("day").tip_pct.mean().loc[order]; ax.scatter(range(4), ms.values, color=RED, marker="D", s=55, zorder=6, label="Group mean")
for i, k in enumerate(order): ax.text(i, 44, f"n={int((d.day==k).sum())}", ha="center", fontsize=9, color="#555")
ax.set_ylim(0, 47); ax.set_xlabel(""); ax.set_ylabel("Tip % of bill"); ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .93))
ax.set_title(f"H4: Tip % by day of week. ANOVA F = {F.statistic:.2f}, p = {F.pvalue:.2f}, eta-squared = {eta2:.3f}", loc="left", fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig("figures/fig3_anova_day.png", dpi=160); plt.close()

# Fig 4: correlations
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
sns.regplot(data=d, x="total_bill", y="tip", ax=ax[0], scatter_kws=dict(s=18, alpha=.55, color=BLUE), line_kws=dict(color=RED))
ax[0].set_xlabel("Total bill ($)"); ax[0].set_ylabel("Tip ($)"); ax[0].set_title(f"Tip $ rises with bill: r = {r_amt.statistic:.2f}", loc="left", fontweight="bold")
sns.regplot(data=d, x="total_bill", y="tip_pct", ax=ax[1], scatter_kws=dict(s=18, alpha=.55, color=ORANGE), line_kws=dict(color=RED))
ax[1].set_ylim(0, 45)
ax[1].set_xlabel("Total bill ($)"); ax[1].set_ylabel("Tip % of bill"); ax[1].set_title(f"But tip % falls with bill: r = {r_pct.statistic:.2f}", loc="left", fontweight="bold")
ax[1].text(.97, .95, "(chart cut at 45%; the 71% outlier\nis included in the statistics)", transform=ax[1].transAxes, ha="right", va="top", fontsize=8.5, color="#555")
plt.tight_layout(); plt.savefig("figures/fig4_correlation.png", dpi=160); plt.close()

# Fig 5: chi-square
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
sh = share.loc[["Weekday", "Weekend"]]
ax[0].bar(sh.index, sh["Small (1-2)"], color=BLUE, label="Small party (1-2)"); ax[0].bar(sh.index, sh["Large (3+)"], bottom=sh["Small (1-2)"], color=ORANGE, label="Large party (3+)")
for i, k in enumerate(sh.index):
    ax[0].text(i, sh.loc[k, "Small (1-2)"]/2, f"{sh.loc[k,'Small (1-2)']:.0f}% (n={ct.loc[k,'Small (1-2)']})", ha="center", color="white", fontweight="bold")
    ax[0].text(i, sh.loc[k, "Small (1-2)"]+sh.loc[k, "Large (3+)"]/2, f"{sh.loc[k,'Large (3+)']:.0f}% (n={ct.loc[k,'Large (3+)']})", ha="center", fontweight="bold")
ax[0].set_ylabel("Share of tables (%)"); ax[0].set_ylim(0, 100); ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2)
ax[0].set_title("Share of tables by party size", loc="left", fontweight="bold")
sns.heatmap(pd.DataFrame(resid, index=ct.index, columns=ct.columns), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-3, vmax=3, cbar_kws={"label": "Standardized residual"}, ax=ax[1], linewidths=1)
ax[1].set_title(f"H6: chi2 = {chi.statistic:.2f}, p = {chi.pvalue:.4f}, V = {V:.2f}", loc="left", fontweight="bold", fontsize=11); ax[1].set_xlabel(""); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("figures/fig5_chisquare.png", dpi=160); plt.close()

# Fig 6: p-values raw vs Holm
fig, ax = plt.subplots(figsize=(10, 4.6)); y = np.arange(len(names))[::-1]
ax.scatter(raw, y, color=BLUE, s=80, label="Raw p-value", zorder=3); ax.scatter(holm, y, color=ORANGE, marker="D", s=70, label="Holm-adjusted p", zorder=3)
ax.axvline(ALPHA, color=RED, ls="--"); ax.text(ALPHA*1.15, len(names)-.4, "alpha = 0.05", color=RED, fontsize=9)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(1e-8, 2); ax.set_xlabel("p-value (log scale)")
ax.legend(frameon=False, loc="lower left"); ax.set_title("Only H5 and H6 survive multiple-testing correction", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig6_pvalues.png", dpi=160); plt.close()

# Fig 7: regression diagnostics
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(mod.fittedvalues, mod.resid, s=16, alpha=.6, color=BLUE); ax[0].axhline(0, color=RED, ls="--")
ax[0].set_xlabel("Fitted tip ($)"); ax[0].set_ylabel("Residual ($)"); ax[0].set_title("Residuals fan out: unequal variance", loc="left", fontweight="bold")
(osm, osr), (slope, icpt, _) = stats.probplot(mod.resid, dist="norm")
ax[1].scatter(osm, osr, s=14, color=BLUE, alpha=.7); ax[1].plot(osm, slope*np.array(osm)+icpt, color=RED)
ax[1].set_xlabel("Theoretical quantiles"); ax[1].set_ylabel("Residual quantiles"); ax[1].set_title("Residual Q-Q plot: heavy right tail", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/fig7_regression_diagnostics.png", dpi=160); plt.close()
print("\nfigures saved")
