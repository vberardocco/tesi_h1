"""Paired Wilcoxon aggregating by day (n = 7): mean of the two files of each held-out day.
Same condition pairs and same Holm correction (per method) as stats_within.csv."""
import csv, itertools, statistics
from collections import defaultdict
R = defaultdict(list)
for x in csv.DictReader(open('results/crosssession/sessions_r2.csv')):
    R[(x['method'], x['condition'], x['day'])].append(float(x['r2']))
day = {k: sum(v)/len(v) for k, v in R.items()}
def wilcoxon_exact(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0: return None
    a = sorted(range(n), key=lambda i: abs(d[i])); ranks = [0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and abs(d[a[j+1]]) == abs(d[a[i]]): j += 1
        for k in range(i, j+1): ranks[a[k]] = (i+j)/2+1
        i = j+1
    wp = sum(r for r, x in zip(ranks, d) if x > 0); tot = sum(ranks)
    stat = min(wp, tot-wp); cnt = 0
    for s in itertools.product([0,1], repeat=n):
        w = sum(r for r, b in zip(ranks, s) if b)
        if min(w, tot-w) <= stat + 1e-12: cnt += 1
    return min(1.0, cnt/2**n)
out = []
for x in csv.DictReader(open('results/crosssession/stats_within.csv')):
    m, a, b = x['method'], x['from'], x['to']
    days = sorted({k[2] for k in day if k[0]==m and k[1]==a} & {k[2] for k in day if k[0]==m and k[1]==b})
    d = [day[(m,b,g)] - day[(m,a,g)] for g in days]
    p = wilcoxon_exact(d)
    out.append(dict(method=m, frm=a, to=b, n_days=len(d), median_diff=statistics.median(d) if d else None,
                    n_better=sum(v>0 for v in d), p=p, p14=x['p'], p14_holm=x['p_holm']))
for m in {o['method'] for o in out}:
    g = sorted([o for o in out if o['method']==m and o['p'] is not None], key=lambda o: o['p'])
    k = len(g); run = 0
    for i, o in enumerate(g):
        run = max(run, min(1.0, (k-i)*o['p'])); o['p_holm'] = run
with open('results/crosssession/stats_within_per_day.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())+['p_holm']); w.writeheader()
    for o in out: o.setdefault('p_holm', None); w.writerow(o)
for o in out:
    print(f"{o['method']:15s} {o['frm']:9s}->{o['to']:13s} n={o['n_days']} med={o['median_diff'] if o['median_diff'] is None else round(o['median_diff'],4):>8} better={o['n_better']} p={o['p'] if o['p'] is None else round(o['p'],4)} holm={o['p_holm'] if o['p_holm'] is None else round(o['p_holm'],4)} | n14 holm={o['p14_holm'][:6]}")
