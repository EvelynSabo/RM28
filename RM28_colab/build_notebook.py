"""Build RM28_colab.ipynb with rm28_verify.py embedded (run after editing the script)."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "rm28_verify.py"), encoding="utf-8").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md(r"""
# RM(2,8): independent recomputation of the bound 88 ≤ ρ(2,8) ≤ 91

This notebook re-runs **from scratch** every finite computation behind the dossier
*“A computer-assisted bound for the covering radius of RM(2,8)”*. It uses none of the
dossier's scripts, binaries, or certificate archives. Its only inputs are the four public classification
catalogues that the paper declares as external completeness assumptions, downloaded from the
authors' sites and recorded by SHA-256.

Each stage writes a JSON **certificate** to `WORKDIR/certificates/`. The last stage checks every certificate
and the numerical inequalities that the proof combines, then writes `MASTER.json`.

| Stage | Paper | What is recomputed | Colab free tier (2 vCPU), rough |
|---|---|---|---|
| `catalogues` | §1, refs | download, SHA-256, class counts (68,443 / 205 / 150,357 / 349) | 1 min |
| `selftest` | §2.1 | two independent profile kernels vs. each other and brute force | 1 min |
| `witness` | §9 | d₂(w) = 88, correlation 80, 28,672 nearest quadratics, two decoders | 1 min |
| `directions` | §7 | 32 three-form types, 14,280 rank identities, 8,160 direction pairs | 1 min |
| `algebra` | §3.2, §8.3 | moment identities, all 32,640 two-point cases, formula spot checks | 1 min |
| `quartic_profiles` | §3.1 | full 2²¹ profiles of all 68,443 quartic classes, explicit nearest quadratics | 30–60 min |
| `point_changes` | Lemma 5.1 | covers at distance 36; all 128 point changes at 38 and 40 | 5–10 min |
| `quartic_pairs` | Lemma 3.1, §4 | all 1,339,637 equal-distance pairs: cardinality, stars, exhaustive search | 10–20 min |
| `degree6_bound` | App. A | all 3,655 odd/odd six-variable pairs, every relative position | 1 min |
| `first56` | §3.3 (six36) | all 987 slices, ~407k pairs, all affine frequency maps | 1–2 h |
| `supports` | §3.2–3.5 | 6-point, 8-point and rank-four support enumerations around all 64,378 quartic classes with d₂ ≥ 30 (~18 CPU-hours) | 15–25 h, resumable; split it |
| `final` | §6–8 | checks every certificate and the derived inequalities, then concludes | 1 min |

**Resuming.** Every stage writes its partial results to `WORKDIR`. Put `WORKDIR` on Google Drive (default),
and simply re-run a cell after a disconnect. Finished work is skipped.
**Splitting the heavy stage (recommended).** Open this notebook several times, possibly on different
accounts sharing one Drive folder. Set `NSHARDS` to the number of copies and give each a different `SHARD`;
with 4 copies each takes roughly 4–6 h. Then run section 6 once. The light stages need to run only once,
before the shards start.

On a 10-core PC the whole pipeline took about 3.5 h of wall time; every certificate was PASS.

Never run Python with `-O`: assertions are part of the proof checks.
"""),
    md("## 1. Settings"),
    code(r"""
USE_DRIVE = True          # keep results on Google Drive so that a disconnect loses nothing
NSHARDS, SHARD = 1, 0     # e.g. NSHARDS = 3 and SHARD = 0, 1, 2 in three notebooks
WORKERS = None            # processes for heavy stages (None = all CPUs)

import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORKDIR = '/content/drive/MyDrive/rm28_work'
else:
    WORKDIR = '/content/rm28_work'
os.makedirs(WORKDIR, exist_ok=True)
W = f'--workers {WORKERS}' if WORKERS else ''
print('WORKDIR =', WORKDIR)
"""),
    md("## 2. The verification program (written to `/content/rm28_verify.py`)"),
    code("%%writefile /content/rm28_verify.py\n" + src),
    md("## 3. Environment"),
    code(r"""
!pip -q install numba numpy
!python -c "import sys, numpy, numba; print(sys.version); print('numpy', numpy.__version__, 'numba', numba.__version__)"
!nproc; free -g | head -2
"""),
    md("## 4. Light stages (all but the support enumerations), about 2–3 hours"),
    code(r"""
!python /content/rm28_verify.py --workdir "{WORKDIR}" {W} light
"""),
    md(r"""
## 5. Heavy stage: support enumerations (resumable, shardable)
Re-run this cell after a disconnect; finished classes are skipped. A progress line with an ETA is printed every 250 classes.
"""),
    code(r"""
!python /content/rm28_verify.py --workdir "{WORKDIR}" {W} --shard {SHARD}/{NSHARDS} supports
"""),
    md("## 6. Final assembly"),
    code(r"""
!python /content/rm28_verify.py --workdir "{WORKDIR}" merge final

import json, glob, os
rows = []
for p in sorted(glob.glob(os.path.join(WORKDIR, 'certificates', '*.json'))):
    c = json.load(open(p))
    rows.append((c['certificate'], c['status'], c.get('wall_seconds', '')))
w = max(len(r[0]) for r in rows)
for name, st, sec in rows:
    print(f'{name:<{w}}  {st:<18} {sec}')
m = json.load(open(os.path.join(WORKDIR, 'certificates', 'MASTER.json')))
print('\nCONCLUSION:', m['results']['conclusion'])
for k, v in m['results']['derived_inequalities'].items():
    if k != 'values':
        print(('  OK  ' if v else '  FAIL'), k)
print('values:', m['results']['derived_inequalities'].get('values'))
for p in m['results']['problems']:
    print('  problem:', p)
"""),
    md("## 7. Download the certificates"),
    code(r"""
import shutil
from google.colab import files
zipbase = '/content/rm28_certificates'
shutil.make_archive(zipbase, 'zip', os.path.join(WORKDIR, 'certificates'))
files.download(zipbase + '.zip')
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(here, "RM28_colab.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote", out)
