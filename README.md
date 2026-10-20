# RM(2,8) bound 88 ≤ ρ(2,8) ≤ 91: independent recomputation for Google Colab

`rm28_verify.py` recomputes, from scratch, every finite enumeration in the dossier
*“A computer-assisted bound for the covering radius of RM(2,8)”* (RM28.tex, 17 Oct 2026). It writes
one JSON certificate per major part of the proof. `RM28_colab.ipynb` is the same program wrapped in a
self-contained Colab notebook: the script is embedded, so no upload is needed.

Nothing from the dossier's `scripts/`, `source/`, `data/` or `certificates/` is used. The only inputs are
the four public catalogues that the paper itself declares as external completeness assumptions. They are
downloaded from the authors' sites, and their SHA-256 digests are recorded:

| File | Source | Content |
|---|---|---|
| `B-3-4-7.dat` | langevin.univ-tln.fr/project/agl7/data | RM(4,7)/RM(2,7), 68,443 classes + stabilizer generators |
| `class-3-6.txt` | langevin.univ-tln.fr/project/agl | RM(6,6)/RM(2,6), 205 classes |
| `class-2-6.txt` | langevin.univ-tln.fr/project/agl | RM(6,6)/RM(1,6), 150,357 classes |
| `classification.txt` | langevin.univ-tln.fr/project/cubics | 349 cubic forms in 9 variables |

## Running it

**Colab:** open `RM28_colab.ipynb` (File → Upload notebook) and run the cells in order. Results go to
`MyDrive/rm28_work`. Re-running a cell after a disconnect resumes where it stopped. To spread the heavy
stage over several Colab sessions at once, set `NSHARDS`/`SHARD` in the settings cell.

**Any machine** (Python ≥ 3.10, `pip install numpy numba`):

```bash
python rm28_verify.py --workdir rm28_work light
```

```bash
python rm28_verify.py --workdir rm28_work supports
```

```bash
python rm28_verify.py --workdir rm28_work final
```

Options: `--workers N` sets the number of processes for `first56`/`supports` (each needs about 0.5 GB RAM).
`--shard i/n` processes every n-th class starting at i. `--limit K` runs a sample, certified as
`PASS_SAMPLE_ONLY`. `merge` rebuilds the support certificates from the shard files.

## Stages and certificates

| Certificate | Paper | Method (independent of the dossier) |
|---|---|---|
| `catalogues` | refs | downloads, digests, parsing, class counts |
| `selftest` | §2.1 | Walsh-split profile kernel vs. direct-popcount kernel vs. brute force |
| `witness` | §9 | d₂ = 88, max correlation 80, 28,672 nearest quadratics; Walsh decoder and direct decoder; explicit nearest quadratic |
| `directions` | §7 | 32 types extracted from the cubic catalogue (radical computed); hyperplane restriction ranks vs. contraction ranks; statements 1 and 2 for all 8,160 type/vector pairs; cubic↔quintic duality checked |
| `algebra` | §3.2, §8.3 | ANF/moment identities; all 32,640 two-point moment cases exhaustively; quartic-distance formula checks |
| `quartic_profiles` | §3.1 | full 2²¹ profiles of all 68,443 classes (two kernels on a subset), an explicit verified nearest quadratic per class, inequalities (q36)–(q40) |
| `point_changes` | Lemma 5.1 | explicit covers at distance 36; all 33·128 and 4·128 point changes profiled |
| `quartic_pairs` | Lemma 3.1 | cardinality, joint star matching, affine hulls, annihilator rank distributions, then an exhaustive search over GL(7,2) × translations, normalized by verified stabilizers and checked against the full profiles |
| `degree6_bound` | App. A | the same pair engine for all 3,655 odd/odd six-variable pairs |
| `first56` | §3.3 | all affine frequency permutations for all slice pairs; every resulting word fully profiled |
| `support6`, `affine8`, `rank4` | §3.2, §3.4 | exact depth-first enumeration of 6-point (sum ≠ 0), 8-point affine-basis and 6-point (sum 0) supports around every quartic class, using a complete quadratic bank (so survivors are exactly the words at the target distance) and orbit reduction by verified stabilizer generators; every survivor is profiled or screened |
| `MASTER` | §6–8 | all certificates PASS + the derived inequalities of Lemmas 6.1/6.2 → 88 ≤ ρ(2,8) ≤ 91 |

Every support search is *exact*. Every quadratic that could constrain a support is in the bank, so the
survivors are exactly the supports reaching the target distance. They are not "retained candidates".
Symmetry reduction uses only stabilizer generators re-verified by truth-table substitution. Reducing by any
subgroup is sound.

## External inputs (not recomputed, as in the paper)

* Completeness of the four catalogues above.
* ρ(1,7) = 56 (Mykkeltveit; Hou), used for the bounds 34+56 and 35+55.
* The algebraic reductions proved in the paper: the split formula, the support/moment arguments, and the
  quartic-distance formula. These are spot-checked in `algebra`.

## Local validation before delivery

The whole pipeline was run end to end on a 10-core / 20-thread Windows laptop (Python 3.12, numpy 2.5,
numba 0.68). **Every certificate is PASS and `MASTER` concludes 88 ≤ ρ(2,8) ≤ 91.** The certificates are in
`local_run_certificates/`. Wall time was about 3.5 h. The heavy `supports` stage used about 29 CPU-hours
under contention, about 18 CPU-hours single-threaded. The results reproduce the paper's numbers:

* witness: distance 88, correlation 80, **28,672** nearest quadratics; both decoders agree.
* quartic tiers: 1,636 / 33 / 4 classes at 36 / 38 / 40; min N36 = 64, distance-38 max M = 54 and min N38 = 896, distance-40 max M = 52.
* directions: 32 types, 14,280 rank identities, 8,160 pairs = 8,032 + 128.
* quartic pairs: distance 36 has 1,337,840 cardinality and 1,221 star exclusions. That is the paper's 1,220
  plus (4, 68385), which the paper settled with a separate argument. The five residual pairs are excluded.
  The search for (4, 4) visits exactly **1,344** maps, the size of the paper's flag orbit.
  Distance 38: 362 + 199. Distance 40: all ten pairs excluded.
* degree-six bound: all 3,655 pairs excluded.
* point changes: all 4,224 + 512 point changes satisfy the strict counting inequalities.
* first56: 983 + 4 slices; exactly **2** maps reach distance 36. Their profile is the paper's table exactly:
  N36 = 512, N56 = 16.
* supports (64,378 classes with d₂ ≥ 30):
  * rank-four: 2,923,088 supports at distance ≥ 36 (the paper retains 2,887,059 under its own
    normalization). 5,199 have M = 56, with **min N36 = 60 and max N56 = 12**. Distance 38 occurs
    23 times, with **max M = 54, min N38 = 608, max N54 = 576**. None reaches 40.
  * degree six: 2,875 six-point words at distance 38, with **max M = 54, min N38 = 968, max N54 = 8**.
    No affine-basis (8-point) support reaches 38. The paper's eight-point "exceptions" are not affine
    bases, which is consistent.
* all 1,636 + 33 quartic histograms equal the profiles archived in the dossier.
* Kernel validation: the support search agrees with brute force on random instances. Symmetry
  normalization equals the full enumeration after closure under the group, on 27 class/kind cases.
