"""Friedman and paired comparisons between methods, aggregating by day (n = 7), same conditions,
same methods and same Holm correction (over all pairs of a condition) as stats_between_*.csv."""
import csv, math, itertools, statistics, sys
sys.path.insert(0, 'scripts')
from collections import defaultdict
from stats_within_per_day import wilcoxon_exact  # noqa (this also re-runs the within script, harmless)
R = defaultdict(list)
for x in csv.DictReader(open('results/crosssession/sessions_r2.csv')):
    R[(x['method'], x['condition'], x['day'])].append(float(x['r2']))
day = {k: sum(v)/len(v) for k, v in R.items()}
def avgrank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2+1
        i = j+1
    return r
def chi2_sf(x, k):
    if k % 2 == 0:
        return math.exp(-x/2)*sum((x/2)**i/math.factorial(i) for i in range(k//2))
    s = math.erfc(math.sqrt(x/2)); t = 0; term = math.sqrt(x)
    for j in range(1, (k-1)//2+1):
        if j > 1: term *= x/(2*j-1)
        t += term
    return s + 2*math.exp(-x/2)/math.sqrt(2*math.pi)*t
out = open('results/crosssession/stats_between_per_day.txt', 'w')
for x in csv.DictReader(open('results/crosssession/stats_between_friedman.csv')):
    cond = x['condition']; M = x['methods'].split(',')
    M = [m if m != 'ridge_channels' else 'ridge_channels' for m in M]
    days = sorted(set.intersection(*[{k[2] for k in day if k[0]==m and k[1]==(cond if m!='ridge_channels' else cond)} for m in M]))
    n, k = len(days), len(M)
    ranks = [avgrank([day[(m, cond, g)] for m in M]) for g in days]
    Rj = [sum(r[j] for r in ranks) for j in range(k)]
    chi2 = 12/(n*k*(k+1))*sum(r*r for r in Rj) - 3*n*(k+1)
    W = chi2/(n*(k-1))
    line = f"{cond}: n={n} k={k} chi2={chi2:.2f} p={chi2_sf(chi2,k-1):.2e} W={W:.2f} | n14: chi2={float(x['chi2']):.1f} p={float(x['p']):.1e}"
    print(line); out.write(line+'\n')
    pairs = []
    for a, b in itertools.combinations(M, 2):
        d = [day[(b, cond, g)]-day[(a, cond, g)] for g in days]
        pairs.append([a, b, statistics.median(d), sum(v > 0 for v in d), wilcoxon_exact(d)])
    srt = sorted(range(len(pairs)), key=lambda i: pairs[i][4]); run = 0
    for r_, i in enumerate(srt):
        run = max(run, min(1.0, (len(pairs)-r_)*pairs[i][4])); pairs[i].append(run)
    for a, b, md, nb, p, ph in pairs:
        if {a, b} & {'PCA', 'LFADS', 'ridge_channels'} and {a, b} & {'PCA', 'LFADS', 'FA', 'GPFA', 'ridge_channels'} == {a, b}:
            l = f"   {a}-{b}: med(b-a)={md:+.3f} b better {nb}/{n} p={p:.4f} holm={ph:.3f}"
            print(l); out.write(l+'\n')
