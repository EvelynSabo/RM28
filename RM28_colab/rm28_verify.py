#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rm28_verify.py -- independent from-scratch recomputation of the finite lemmas
behind the dossier "A computer-assisted bound for the covering radius of
RM(2,8):  88 <= rho(2,8) <= 91"  (RM28.tex, 17 Oct 2026).

Nothing from the dossier's scripts, binaries, or certificate archives is used.
The only inputs are the public classification catalogues that the paper itself
declares to be external completeness assumptions; they are downloaded from the
authors' web sites and their SHA-256 digests are recorded in every certificate.

Every stage writes a JSON certificate under  <workdir>/certificates/ .
Heavy stages are sharded and resumable, so they can be spread over several
Google Colab sessions (results are kept on Google Drive if mounted).

Run  `python rm28_verify.py --help`  for the stage list.  In Colab, see the
companion notebook RM28_colab.ipynb.

All arithmetic is exact (integers / GF(2)).  Assertions are proof checks:
never run with `python -O`.
"""

import argparse
import datetime
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import platform
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover
    sys.exit("numba is required:  pip install numba")

if not __debug__:
    sys.exit("Do not run with python -O: assertions are part of the proof checks.")

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# 0. Paths, configuration, certificate I/O
# ---------------------------------------------------------------------------

CATALOGUES = {
    # name: (list of URLs, expected size or None, description)
    "B-3-4-7.dat": (
        ["http://langevin.univ-tln.fr/project/agl7/data/B-3-4-7.dat",
         "https://langevin.univ-tln.fr/project/agl7/data/B-3-4-7.dat"],
        "Gillot-Langevin: RM(4,7)/RM(2,7) under AGL(7,2), 68,443 classes with stabilizer generators"),
    "class-3-6.txt": (
        ["http://langevin.univ-tln.fr/project/agl/class-3-6.txt",
         "https://raw.githubusercontent.com/drmichko/radii/main/class-3-6.txt"],
        "Langevin: six-variable functions modulo RM(2,6) under AGL(6,2), 205 classes"),
    "class-2-6.txt": (
        ["http://langevin.univ-tln.fr/project/agl/class-2-6.txt",
         "https://langevin.univ-tln.fr/project/agl/class-2-6.txt"],
        "Langevin: six-variable functions modulo RM(1,6) under AGL(6,2), 150,357 classes"),
    "classification.txt": (
        ["https://langevin.univ-tln.fr/project/cubics/classification.txt",
         "http://langevin.univ-tln.fr/project/cubics/classification.txt"],
        "Brier-Langevin: 349 cubic forms in nine variables modulo RM(2,9) under GL(9,2)"),
}

# SHA-256 digests of the copies used by the dossier (informational only: a
# mismatch is reported but is not an error; the catalogue's content is
# re-validated by class counts below).
DOSSIER_SHA256 = {
    "B-3-4-7.dat": "c57fbc0d001b6f61d1389147a1b6cd1b395013d0ae2006dcb8876e49c9c29e41",
    "class-2-6.txt": "8174e08ac8e974e25a662f482be9e6e7969910d83bc3ce22e57eaf9f1d75367d",
}


class Ctx:
    """Run context: directories, thread count, catalogue cache."""

    def __init__(self, workdir, threads=None, quiet=False):
        self.workdir = os.path.abspath(workdir)
        self.data = os.path.join(self.workdir, "data")
        self.certs = os.path.join(self.workdir, "certificates")
        self.work = os.path.join(self.workdir, "work")
        for d in (self.data, self.certs, self.work):
            os.makedirs(d, exist_ok=True)
        if threads:
            numba.set_num_threads(int(threads))
        self.quiet = quiet
        self._cache = {}

    def log(self, *a):
        if not self.quiet:
            print(time.strftime("[%H:%M:%S]"), *a, flush=True)

    def cert_path(self, name):
        return os.path.join(self.certs, name + ".json")

    def stage_dir(self, name):
        d = os.path.join(self.work, name)
        os.makedirs(d, exist_ok=True)
        return d


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def code_sha256():
    try:
        return sha256_file(os.path.abspath(__file__))
    except Exception:
        return None


def environment():
    return {
        "script": os.path.basename(__file__),
        "script_version": VERSION,
        "script_sha256": code_sha256(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numba_threads": numba.get_num_threads(),
    }


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


def write_json_gz(path, obj):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump(obj, fh, default=_json_default)
    os.replace(tmp, path)


def open_append_clean(path):
    """Open a JSON-lines file for appending, first terminating a line that an
    interruption may have left incomplete."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with open(path, "a") as fh:
                fh.write("\n")
    return open(path, "a")


def read_json(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def write_certificate(ctx, name, claim, status, results, inputs=None, extra=None,
                      started=None):
    cert = {
        "certificate": name,
        "claim": claim,
        "status": status,
        "results": results,
        "inputs": inputs or {},
        "environment": environment(),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if started is not None:
        cert["wall_seconds"] = round(time.time() - started, 2)
    if extra:
        cert.update(extra)
    write_json(ctx.cert_path(name), cert)
    ctx.log(f"certificate {name}: {status}")
    return cert


class CheckFailure(Exception):
    pass


def check(cond, msg):
    """Proof check that survives `python -O` (which is refused anyway)."""
    if not cond:
        raise CheckFailure(msg)


# ---------------------------------------------------------------------------
# 1. Boolean functions: truth tables, ANF, quadratic coefficient spaces
#    Convention: variable i (letter chr(97+i)) is bit i of the input index;
#    a truth table is a Python int whose bit x is f(x).
# ---------------------------------------------------------------------------

LETTERS = "abcdefghi"
M64 = (1 << 64) - 1


def low_masks(n):
    """LOW[i] = int with bit x set iff bit i of x is 0 (x < 2^n)."""
    N = 1 << n
    return [sum(1 << x for x in range(N) if not (x >> i) & 1) for i in range(n)]


_LOW = {n: low_masks(n) for n in range(1, 10)}


def mobius(t, n):
    """ANF <-> truth table (the binary Moebius transform is an involution)."""
    for i in range(n):
        t ^= (t & _LOW[n][i]) << (1 << i)
    return t


def monomial_truth(mask, n):
    t = 0
    for x in range(1 << n):
        if x & mask == mask:
            t |= 1 << x
    return t


def parse_anf(s):
    """'abc+bde' -> list of monomial masks.  '0' or '' -> []; '1' -> [0]."""
    s = s.strip().replace(" ", "")
    if s in ("", "0"):
        return []
    out = []
    for term in s.split("+"):
        if term == "1":
            out.append(0)
            continue
        m = 0
        for ch in term:
            k = ord(ch) - 97
            assert 0 <= k < 9, s
            assert not (m >> k) & 1, s
            m |= 1 << k
        out.append(m)
    return out


def anf_int(monos):
    a = 0
    for m in monos:
        a ^= 1 << m
    return a


def anf_string(a, n):
    terms = []
    for m in range(1 << n):
        if (a >> m) & 1:
            terms.append("".join(LETTERS[i] for i in range(n) if (m >> i) & 1) or "1")
    return "+".join(terms) if terms else "0"


def truth_from_anf_string(s, n):
    return mobius(anf_int(parse_anf(s)), n)


def degree(t, n):
    a = mobius(t, n)
    d = -1
    m = 0
    while a:
        if a & 1:
            d = max(d, bin(m).count("1"))
        a >>= 1
        m += 1
    return d


def weight(t):
    return bin(t).count("1")


def pairs(n):
    """Colex order of pairs: (0,1),(0,2),(1,2),(0,3),...  Nested: pairs(n) is
    a prefix of pairs(n+1).  A homogeneous quadratic on n variables is an int
    whose bit k is the coefficient of x_i x_j for (i,j) = pairs(n)[k]."""
    return [(i, j) for j in range(n) for i in range(j)]


PAIRS = {n: pairs(n) for n in range(2, 10)}


def quad_truth(Q, n):
    """Truth table (int) of the homogeneous quadratic with coefficient int Q."""
    t = 0
    for k, (i, j) in enumerate(PAIRS[n]):
        if (Q >> k) & 1:
            t ^= monomial_truth((1 << i) | (1 << j), n)
    return t


def linear_truth(a, n):
    t = 0
    for x in range(1 << n):
        if bin(a & x).count("1") & 1:
            t |= 1 << x
    return t


def alt_matrix_from_Q(Q, n):
    """Symmetric/alternating 0/1 matrix (list of row bitmasks) of Q."""
    rows = [0] * n
    for k, (i, j) in enumerate(PAIRS[n]):
        if (Q >> k) & 1:
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
    return rows


def Q_from_alt_matrix(rows, n):
    Q = 0
    for k, (i, j) in enumerate(PAIRS[n]):
        if (rows[i] >> j) & 1:
            Q |= 1 << k
    return Q


def gf2_rank(vecs):
    """Rank of a list of int bit-vectors over GF(2)."""
    basis = []
    for v in vecs:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def gf2_basis(vecs):
    """Reduced echelon basis {pivot: vector} (pivot = highest set bit)."""
    piv = {}
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                break
    return piv


def gf2_reduce(v, piv):
    while v:
        h = v.bit_length() - 1
        if h in piv:
            v ^= piv[h]
        else:
            return v
    return 0


def gf2_span(vecs):
    out = {0}
    for b in vecs:
        if b not in out:
            out |= {x ^ b for x in out}
    return out


def apply_linear(rows, x):
    """y = A x where A is given by its columns?  No: rows[i] is the bitmask of
    row i, so bit i of y is <rows[i], x>."""
    y = 0
    for i, r in enumerate(rows):
        if bin(r & x).count("1") & 1:
            y |= 1 << i
    return y


def compose_truth_affine(t, n, rows, c):
    """Return the truth table of x -> t(A x + c)  (A given by row masks)."""
    out = 0
    for x in range(1 << n):
        y = apply_linear(rows, x) ^ c
        if (t >> y) & 1:
            out |= 1 << x
    return out


def strip_low_degree(t, n, d):
    """Remove all ANF terms of degree <= d (canonical rep modulo RM(d,n))."""
    a = mobius(t, n)
    for m in range(1 << n):
        if (a >> m) & 1 and bin(m).count("1") <= d:
            a ^= 1 << m
    return mobius(a, n)


def invert_gf2_matrix(rows, n):
    """Inverse of an n x n GF(2) matrix given by row masks (None if singular)."""
    aug = [(rows[i], 1 << i) for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if (aug[r][0] >> col) & 1:
                piv = r
                break
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col and (aug[r][0] >> col) & 1:
                aug[r] = (aug[r][0] ^ aug[col][0], aug[r][1] ^ aug[col][1])
    return [aug[i][1] for i in range(n)]


def mat_mul_rows(A, B, n):
    """(A B) for row-mask matrices: row i of AB = sum_k A[i][k] B[k]."""
    out = []
    for i in range(n):
        r = 0
        for k in range(n):
            if (A[i] >> k) & 1:
                r ^= B[k]
        out.append(r)
    return out


def transpose_rows(A, n):
    return [sum(((A[i] >> j) & 1) << i for i in range(n)) for j in range(n)]


# ---------------------------------------------------------------------------
# 2. Catalogues: download, digest, parse
# ---------------------------------------------------------------------------

def fetch_catalogue(ctx, name):
    path = os.path.join(ctx.data, name)
    if not os.path.exists(path):
        urls, desc = CATALOGUES[name]
        last = None
        for url in urls:
            try:
                ctx.log(f"downloading {name} from {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "rm28-verify"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                with open(path + ".part", "wb") as fh:
                    fh.write(data)
                os.replace(path + ".part", path)
                break
            except Exception as e:  # try next mirror
                last = e
        else:
            raise RuntimeError(
                f"Could not download {name} ({last}). Upload it manually to {ctx.data}")
    digest = sha256_file(path)
    info = {"file": name, "bytes": os.path.getsize(path), "sha256": digest,
            "source": CATALOGUES[name][0][0], "description": CATALOGUES[name][1]}
    if name in DOSSIER_SHA256:
        info["matches_dossier_copy"] = (digest == DOSSIER_SHA256[name])
    return path, info


def parse_class_file(text, n):
    """Parse Langevin's 'anf=/hdr=' class files with stabilizer generators.

    Returns a list of dicts {anf, gens: [(rows, trans)], stab}.  Each generator
    line is '[r_1 ... r_n]t' with n-bit binary words; the interpretation of
    these words is established and *verified* by `stabilizer_generators`."""
    entries = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("anf=") or line.startswith("hdr="):
            cur = {"anf": line[4:].strip(), "gens": [], "stab": None}
            entries.append(cur)
        elif line.startswith("["):
            m = re.match(r"\[([01 ]+)\]\s*([01]+)", line)
            assert m, line
            rows = m.group(1).split()
            assert len(rows) == n and all(len(r) == n for r in rows), line
            assert len(m.group(2)) == n, line
            cur["gens"].append((rows, m.group(2)))
        elif line.startswith("stabSize=") or line.startswith("fix="):
            cur["stab"] = int(line.split("=", 1)[1])
        else:
            raise ValueError("unexpected catalogue line: " + line[:80])
    return entries


def gen_to_affine(rows, trans, n, convention):
    """Convert catalogue words to (row masks A, translation c) for x -> A x + c.

    convention = (transpose, reverse): bit order of the words (reverse=False:
    first character is variable a) and whether the words are columns."""
    transpose, reverse = convention

    def word(s):
        s = s[::-1] if reverse else s
        return sum(1 << i for i, ch in enumerate(s) if ch == "1")

    M = [word(r) for r in rows]
    if transpose:
        M = transpose_rows(M, n)
    return M, word(trans)


def find_generator_convention(entries, n, truths, modulo_degree, samples=300):
    """Determine the generator encoding empirically and return it.  A
    convention is accepted only if every generator of the sampled classes
    fixes its class representative modulo RM(modulo_degree, n) (this property
    is re-verified for every generator that is later used)."""
    best = None
    idx = [i for i in range(len(entries)) if entries[i]["gens"]][:samples]
    for transpose in (False, True):
        for reverse in (False, True):
            for inverse in (False, True):
                ok = 0
                tot = 0
                for i in idx:
                    t = truths[i]
                    base = strip_low_degree(t, n, modulo_degree)
                    for rows, tr in entries[i]["gens"][:3]:
                        A, c = gen_to_affine(rows, tr, n, (transpose, reverse))
                        if invert_gf2_matrix(A, n) is None:
                            continue
                        if inverse:
                            Ai = invert_gf2_matrix(A, n)
                            A, c = Ai, apply_linear(Ai, c)
                        tot += 1
                        img = strip_low_degree(compose_truth_affine(t, n, A, c), n, modulo_degree)
                        ok += img == base
                if tot and ok == tot:
                    return (transpose, reverse, inverse)
                if best is None or ok > best[0]:
                    best = (ok, tot, (transpose, reverse, inverse))
    raise RuntimeError(f"no generator convention verifies (best {best})")


def load_quartic_catalogue(ctx):
    """68,443 representatives of RM(4,7)/RM(2,7) (Gillot-Langevin)."""
    if "quartics" in ctx._cache:
        return ctx._cache["quartics"]
    path, info = fetch_catalogue(ctx, "B-3-4-7.dat")
    with open(path) as fh:
        text = fh.read()
    entries = parse_class_file(text, 7)
    check(len(entries) == 68443, f"quartic catalogue has {len(entries)} classes, expected 68443")
    truths = [truth_from_anf_string(e["anf"], 7) for e in entries]
    for t in truths:
        check(degree(t, 7) <= 4, "catalogue entry of degree > 4")
    info["classes"] = len(entries)
    info["with_stabilizer_generators"] = sum(1 for e in entries if e["gens"])
    ctx._cache["quartics"] = (entries, truths, info)
    return entries, truths, info


def truths_to_lohi(truths):
    lo = np.array([t & M64 for t in truths], dtype=np.uint64)
    hi = np.array([(t >> 64) & M64 for t in truths], dtype=np.uint64)
    return lo, hi


# ---------------------------------------------------------------------------
# 3. numba kernels
# ---------------------------------------------------------------------------

_U1 = np.uint64(1)


def make_tables():
    """Truth tables (uint64, six variables) of the 2^15 homogeneous quadratics
    and the 64 linear functions."""
    mono = [monomial_truth((1 << i) | (1 << j), 6) for (i, j) in PAIRS[6]]
    QT6 = np.zeros(1 << 15, dtype=np.uint64)
    idx = np.arange(1 << 15)
    for k in range(15):
        QT6 ^= np.where((idx >> k) & 1, np.uint64(mono[k]), np.uint64(0))
    LIN6 = np.array([linear_truth(a, 6) for a in range(64)], dtype=np.uint64)
    return QT6, LIN6


QT6, LIN6 = make_tables()


@njit(cache=True, inline="always")
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, inline="always")
def walsh6(g, W):
    """Signed Walsh spectrum W[a] = sum_x (-1)^{g(x)+a.x} of a 6-variable
    function given as a uint64 truth table (fast Walsh-Hadamard transform)."""
    for x in range(64):
        W[x] = 1 - 2 * np.int32((g >> np.uint64(x)) & np.uint64(1))
    h = 1
    while h < 64:
        for b in range(0, 64, 2 * h):
            for j in range(b, b + h):
                u = W[j]
                v = W[j + h]
                W[j] = u + v
                W[j + h] = u - v
        h *= 2


@njit(cache=True)
def profile7_walsh(lo, hi, QT6, out):
    """Method W.  out[Q] = d(f+Q, RM(1,7)) for all 2^21 homogeneous quadratics
    Q = q + x_6*lambda (q: bits 0..14, lambda: bits 15..20), computed as
    64 - max_a(|W_{f0+q}(a)| + |W_{f1+q}(a+lambda)|)/2."""
    W0 = np.empty(64, np.int32)
    W1 = np.empty(64, np.int32)
    for q in range(32768):
        qt = QT6[q]
        walsh6(lo ^ qt, W0)
        walsh6(hi ^ qt, W1)
        for a in range(64):
            W0[a] = abs(W0[a])
            W1[a] = abs(W1[a])
        for lam in range(64):
            m = 0
            for a in range(64):
                s = W0[a] + W1[a ^ lam]
                if s > m:
                    m = s
            out[q | (lam << 15)] = 64 - m // 2


@njit(cache=True)
def profile7_direct(lo, hi, QT6, LIN6, out):
    """Method D (independent of Walsh transforms).  out[Q] = min over the
    affine functions of the Hamming weight of f+Q+affine, using popcounts of
    each half against the 64 linear functions and both constants."""
    d0 = np.empty(64, np.int32)
    d1 = np.empty(64, np.int32)
    for q in range(32768):
        g0 = lo ^ QT6[q]
        g1 = hi ^ QT6[q]
        for a in range(64):
            w = popcount64(g0 ^ LIN6[a])
            d0[a] = min(w, 64 - w)
            w = popcount64(g1 ^ LIN6[a])
            d1[a] = min(w, 64 - w)
        for lam in range(64):
            m = 1000
            for a in range(64):
                s = d0[a] + d1[a ^ lam]
                if s < m:
                    m = s
            out[q | (lam << 15)] = m


@njit(cache=True)
def hist7_walsh(lo, hi, QT6, hist):
    """Histogram of the full profile (hist[j] = N_j), method W.  Returns the
    first Q attaining the minimum."""
    W0 = np.empty(64, np.int32)
    W1 = np.empty(64, np.int32)
    for j in range(hist.shape[0]):
        hist[j] = 0
    best = 1000
    bestQ = -1
    for q in range(32768):
        qt = QT6[q]
        walsh6(lo ^ qt, W0)
        walsh6(hi ^ qt, W1)
        for a in range(64):
            W0[a] = abs(W0[a])
            W1[a] = abs(W1[a])
        for lam in range(64):
            m = 0
            for a in range(64):
                s = W0[a] + W1[a ^ lam]
                if s > m:
                    m = s
            c = 64 - m // 2
            hist[c] += 1
            if c < best:
                best = c
                bestQ = q | (lam << 15)
    return bestQ


@njit(cache=True, parallel=True)
def batch_hist7(LO, HI, QT6, HIST, ARG):
    for i in prange(LO.shape[0]):
        ARG[i] = hist7_walsh(LO[i], HI[i], QT6, HIST[i])


@njit(cache=True, parallel=True)
def batch_profile7_direct(LO, HI, QT6, LIN6, HIST):
    """Histograms by method D (used to cross-check method W)."""
    for i in prange(LO.shape[0]):
        out = np.empty(1 << 21, np.uint8)
        profile7_direct(LO[i], HI[i], QT6, LIN6, out)
        for j in range(HIST.shape[1]):
            HIST[i, j] = 0
        for Q in range(1 << 21):
            HIST[i, out[Q]] += 1


@njit(cache=True)
def candidate_lambda_masks(P, threshold):
    """mask[q] = bitmask of the lambdas with P[q | lambda<<15] >= threshold."""
    mask = np.zeros(32768, np.uint64)
    for Q in range(1 << 21):
        if P[Q] >= threshold:
            mask[Q & 32767] |= np.uint64(1) << np.uint64(Q >> 15)
    return mask


@njit(cache=True)
def count_cost_at_least(lo, hi, QT6, cand, level):
    """Number of Q with d(f+Q, RM(1,7)) >= level, scanning only the Q flagged
    in cand (cand[q] bit lambda).  Sound when every Q outside cand is known to
    have cost < level (e.g. |C_f - C_u| <= |E| and C_u(Q) < level - |E|)."""
    W0 = np.empty(64, np.int32)
    W1 = np.empty(64, np.int32)
    lim = 2 * (64 - level)          # need max_a |W0(a)|+|W1(a^lam)| <= lim
    cnt = 0
    for q in range(32768):
        m = cand[q]
        if m == 0:
            continue
        qt = QT6[q]
        walsh6(lo ^ qt, W0)
        mx0 = 0
        for a in range(64):
            v = abs(W0[a])
            W0[a] = v
            if v > mx0:
                mx0 = v
        if mx0 > lim:
            continue
        walsh6(hi ^ qt, W1)
        mx1 = 0
        for a in range(64):
            v = abs(W1[a])
            W1[a] = v
            if v > mx1:
                mx1 = v
        if mx1 > lim:
            continue
        for lam in range(64):
            if not (m >> np.uint64(lam)) & np.uint64(1):
                continue
            ok = True
            for a in range(64):
                if W0[a] + W1[a ^ lam] > lim:
                    ok = False
                    break
            if ok:
                cnt += 1
    return cnt


def profile7(t, method="W"):
    """Full profile (np.uint8[2^21]) of a seven-variable truth table int."""
    out = np.empty(1 << 21, np.uint8)
    lo, hi = np.uint64(t & M64), np.uint64((t >> 64) & M64)
    if method == "W":
        profile7_walsh(lo, hi, QT6, out)
    else:
        profile7_direct(lo, hi, QT6, LIN6, out)
    return out


def hist_from_profile(P, size=65):
    return np.bincount(P, minlength=size)[:size]


def nearest_affine(t_plus_q, n, target):
    """An explicit affine function l (as (linear mask, const)) with
    wt(t_plus_q + l) == target, or None."""
    N = 1 << n
    for a in range(N):
        w = weight(t_plus_q ^ linear_truth(a, n))
        if w == target:
            return a, 0
        if N - w == target:
            return a, 1
    return None


LIN7_LO = np.array([linear_truth(a, 7) & M64 for a in range(128)], dtype=np.uint64)
LIN7_HI = np.array([linear_truth(a, 7) >> 64 for a in range(128)], dtype=np.uint64)


# ---------------------------------------------------------------------------
# 4. Exact support enumeration around a quartic
#
#    For a quartic u and a quadratic q put B_q = supp(u+q), w_q = |B_q|.  For
#    a k-point set E,   wt(u + 1_E + q) = w_q + k - 2|E cap B_q|.
#    Hence d_2(u + 1_E) >= D  iff  |E cap B_q| <= (w_q + k - D)/2 for EVERY
#    quadratic q.  Quadratics with w_q >= D + k - 1 never constrain E, so the
#    bank of all (Q, affine) with w_q <= D + k - 2 gives an EXACT test.  The
#    depth-first search enforces the strong part of the bank (small bounds)
#    while choosing points; every survivor is then tested against the whole
#    bank, so the final list is exactly {E : d_2(u+1_E) >= D}.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bank_kernel(lo, hi, P, wmax, QT6, LIN6, out_lo, out_hi, out_w, out_Q):
    """All quadratics q = Q + affine with wt(u+q) <= wmax.  Returns count
    (or -1 if the output arrays are too small)."""
    n = 0
    cap = out_lo.shape[0]
    for Q in range(1 << 21):
        if P[Q] > wmax:
            continue
        q = Q & 32767
        lam = Q >> 15
        glo = lo ^ QT6[q]
        ghi = hi ^ QT6[q] ^ LIN6[lam]
        for a in range(128):
            l_lo = LIN6[a & 63]
            l_hi = LIN6[a & 63]
            if a >= 64:
                l_hi = ~l_hi
            elo = glo ^ l_lo
            ehi = ghi ^ l_hi
            w = popcount64(elo) + popcount64(ehi)
            if w <= wmax:
                if n >= cap:
                    return -1
                out_lo[n] = elo
                out_hi[n] = ehi
                out_w[n] = w
                out_Q[n] = Q
                n += 1
            if 128 - w <= wmax:
                if n >= cap:
                    return -1
                out_lo[n] = ~elo
                out_hi[n] = ~ehi
                out_w[n] = 128 - w
                out_Q[n] = Q
                n += 1
    return n


def quartic_bank(t, wmax, P=None):
    """Bank of all quadratics q (with affine part) such that wt(t+q) <= wmax,
    as arrays (B_lo, B_hi, w, Q)."""
    if P is None:
        P = profile7(t)
    lo, hi = np.uint64(t & M64), np.uint64(t >> 64)
    est = int((P <= wmax).sum())
    cap = max(1024, 4 * est + 1024)
    while True:
        out_lo = np.empty(cap, np.uint64)
        out_hi = np.empty(cap, np.uint64)
        out_w = np.empty(cap, np.int16)
        out_Q = np.empty(cap, np.int32)
        n = _bank_kernel(lo, hi, P, wmax, QT6, LIN6, out_lo, out_hi, out_w, out_Q)
        if n >= 0:
            return out_lo[:n], out_hi[:n], out_w[:n], out_Q[:n]
        cap *= 4


@njit(cache=True, inline="always")
def _bit_test(lo, hi, p):
    if p < 64:
        return (lo >> np.uint64(p)) & np.uint64(1)
    return (hi >> np.uint64(p - 64)) & np.uint64(1)


@njit(cache=True)
def _greater_mask(p):
    """(lo, hi) mask of points > p."""
    if p < 63:
        lo = ~np.uint64(0) << np.uint64(p + 1)
        return lo, ~np.uint64(0)
    if p == 63:
        return np.uint64(0), ~np.uint64(0)
    if p == 127:
        return np.uint64(0), np.uint64(0)
    return np.uint64(0), ~np.uint64(0) << np.uint64(p - 63)


@njit(cache=True, inline="always")
def _lowest(lo, hi):
    if lo:
        x = lo & (~lo + np.uint64(1))
        k = 0
        while x > np.uint64(1):
            x >>= np.uint64(1)
            k += 1
        return k
    x = hi & (~hi + np.uint64(1))
    k = 64
    while x > np.uint64(1):
        x >>= np.uint64(1)
        k += 1
    return k


@njit(cache=True, inline="always")
def _bit(p):
    if p < 64:
        return np.uint64(1) << np.uint64(p), np.uint64(0)
    return np.uint64(0), np.uint64(1) << np.uint64(p - 64)


@njit(cache=True)
def _translate_set(alo, ahi, v):
    """{x ^ v : x in A} for a 128-bit set A."""
    blo = np.uint64(0)
    bhi = np.uint64(0)
    for x in range(128):
        if _bit_test(alo, ahi, x):
            y = x ^ v
            if y < 64:
                blo |= np.uint64(1) << np.uint64(y)
            else:
                bhi |= np.uint64(1) << np.uint64(y - 64)
    return blo, bhi


@njit(cache=True)
def support_dfs(k, mode, B_lo, B_hi, bnd, ptr, idx, adj_lo, adj_hi,
                F0_lo, F0_hi, base_lo, base_hi, fixed_pts, out, max_out, stats):
    """Enumerate the k-point sets E that contain the points `fixed_pts`, whose
    other points lie in `base` (chosen in increasing order), and such that
    |E cap B_j| <= bnd[j] for every DFS bank item j.

    mode 0: no extra condition;  mode 1: sum(E) == 0 (last point determined
    by the others);  mode 2: sum(E) != 0;  mode 3: E affinely independent.
    Bank items of bound 0 are folded into the forbidden mask F0, items of
    bound 1 into the conflict masks adj[p], items of bound >= 2 are listed per
    point (CSR ptr/idx) and counted with popcounts.
    Writes survivors as rows of `out`; returns their number (-1 on overflow).
    stats[0] += number of DFS nodes."""
    P = np.empty(k + 1, np.int64)
    S_lo = np.zeros(k + 2, np.uint64)
    S_hi = np.zeros(k + 2, np.uint64)
    F_lo = np.zeros(k + 2, np.uint64)
    F_hi = np.zeros(k + 2, np.uint64)
    C_lo = np.zeros(k + 2, np.uint64)
    C_hi = np.zeros(k + 2, np.uint64)
    SP_lo = np.zeros(k + 2, np.uint64)
    SP_hi = np.zeros(k + 2, np.uint64)
    X = np.zeros(k + 2, np.int64)
    nout = 0
    nodes = 0
    F_lo[0] = F0_lo
    F_hi[0] = F0_hi
    nf = fixed_pts.shape[0]
    # --- pre-selected points (symmetry normalisation prefix) ---
    for lv in range(nf):
        p = fixed_pts[lv]
        if _bit_test(F_lo[lv], F_hi[lv], p) or _bit_test(S_lo[lv], S_hi[lv], p):
            return 0
        if mode == 3 and lv > 0 and _bit_test(SP_lo[lv], SP_hi[lv], p):
            return 0
        flo = F_lo[lv] | adj_lo[p]
        fhi = F_hi[lv] | adj_hi[p]
        slo = S_lo[lv]
        shi = S_hi[lv]
        for t in range(ptr[p], ptr[p + 1]):
            j = idx[t]
            c = popcount64(B_lo[j] & slo) + popcount64(B_hi[j] & shi) + 1
            if c >= bnd[j]:
                flo |= B_lo[j]
                fhi |= B_hi[j]
        P[lv] = p
        blo, bhi = _bit(p)
        S_lo[lv + 1] = slo | blo
        S_hi[lv + 1] = shi | bhi
        F_lo[lv + 1] = flo
        F_hi[lv + 1] = fhi
        X[lv + 1] = X[lv] ^ p
        if lv == 0:
            SP_lo[1] = blo
            SP_hi[1] = bhi
        else:
            tlo, thi = _translate_set(SP_lo[lv], SP_hi[lv], p ^ P[0])
            SP_lo[lv + 1] = SP_lo[lv] | tlo
            SP_hi[lv + 1] = SP_hi[lv] | thi
    lvl0 = nf
    C_lo[lvl0] = base_lo & ~F_lo[lvl0] & ~S_lo[lvl0]
    C_hi[lvl0] = base_hi & ~F_hi[lvl0] & ~S_hi[lvl0]
    if mode == 3 and nf > 0:
        C_lo[lvl0] &= ~SP_lo[lvl0]
        C_hi[lvl0] &= ~SP_hi[lvl0]
    lvl = lvl0
    while lvl >= lvl0:
        if mode == 1 and lvl == k - 1:
            # the last point is determined by the zero-sum condition
            p = X[lvl]
            ok = True
            if lvl > lvl0 and p <= P[lvl - 1]:
                ok = False
            if ok and not _bit_test(base_lo, base_hi, p):
                ok = False
            if ok and (_bit_test(F_lo[lvl], F_hi[lvl], p) or _bit_test(S_lo[lvl], S_hi[lvl], p)):
                ok = False
            if ok:
                if nout >= max_out:
                    return -1
                for i in range(lvl):
                    out[nout, i] = P[i]
                out[nout, lvl] = p
                nout += 1
            lvl -= 1
            continue
        if lvl == k:
            if not (mode == 2 and X[lvl] == 0):
                if nout >= max_out:
                    return -1
                for i in range(k):
                    out[nout, i] = P[i]
                nout += 1
            lvl -= 1
            continue
        need = k - lvl
        if mode == 1:
            need = k - 1 - lvl
        if popcount64(C_lo[lvl]) + popcount64(C_hi[lvl]) < need:
            lvl -= 1
            continue
        p = _lowest(C_lo[lvl], C_hi[lvl])
        blo, bhi = _bit(p)
        C_lo[lvl] &= ~blo
        C_hi[lvl] &= ~bhi
        nodes += 1
        flo = F_lo[lvl] | adj_lo[p]
        fhi = F_hi[lvl] | adj_hi[p]
        slo = S_lo[lvl]
        shi = S_hi[lvl]
        for t in range(ptr[p], ptr[p + 1]):
            j = idx[t]
            c = popcount64(B_lo[j] & slo) + popcount64(B_hi[j] & shi) + 1
            if c >= bnd[j]:
                flo |= B_lo[j]
                fhi |= B_hi[j]
        P[lvl] = p
        nl = lvl + 1
        S_lo[nl] = slo | blo
        S_hi[nl] = shi | bhi
        F_lo[nl] = flo
        F_hi[nl] = fhi
        X[nl] = X[lvl] ^ p
        g_lo, g_hi = _greater_mask(p)
        clo = C_lo[lvl] & ~flo & g_lo
        chi = C_hi[lvl] & ~fhi & g_hi
        if mode == 3:
            if nl == 1:
                SP_lo[1] = blo
                SP_hi[1] = bhi
            else:
                tlo, thi = _translate_set(SP_lo[lvl], SP_hi[lvl], p ^ P[0])
                SP_lo[nl] = SP_lo[lvl] | tlo
                SP_hi[nl] = SP_hi[lvl] | thi
            clo &= ~SP_lo[nl]
            chi &= ~SP_hi[nl]
        C_lo[nl] = clo
        C_hi[nl] = chi
        lvl = nl
    stats[0] += nodes
    return nout


def build_dfs_bank(B_lo, B_hi, w, k, D, bmax):
    """Split a bank into the DFS structures.  bound_j = floor((w_j + k - D)/2);
    items with bound >= k are useless; only items with bound <= bmax are used
    inside the search (the rest are applied to survivors afterwards)."""
    bound = (w.astype(np.int64) + k - D) // 2
    check(bool((bound >= 0).all()), "bank contains a quadratic closer than allowed")
    use = bound <= bmax
    F0_lo = np.uint64(0)
    F0_hi = np.uint64(0)
    sel0 = np.nonzero(bound == 0)[0]
    if len(sel0):
        F0_lo = np.bitwise_or.reduce(B_lo[sel0])
        F0_hi = np.bitwise_or.reduce(B_hi[sel0])
    adj_lo = np.zeros(128, np.uint64)
    adj_hi = np.zeros(128, np.uint64)
    sel1 = np.nonzero(bound == 1)[0]
    selm = np.nonzero(use & (bound >= 2))[0]
    _fill_adj(B_lo, B_hi, sel1.astype(np.int64), adj_lo, adj_hi)
    # per-point CSR lists of the multi-bound items (2 <= bound <= bmax)
    ptr, idx = _csr_points(B_lo, B_hi, selm.astype(np.int64))
    return (bound.astype(np.int16), ptr, idx, adj_lo, adj_hi, F0_lo, F0_hi,
            int(len(sel0)), int(len(sel1)), int(len(selm)))


@njit(cache=True)
def _fill_adj(B_lo, B_hi, sel, adj_lo, adj_hi):
    for t in range(sel.shape[0]):
        j = sel[t]
        for p in range(128):
            if _bit_test(B_lo[j], B_hi[j], p):
                adj_lo[p] |= B_lo[j]
                adj_hi[p] |= B_hi[j]
    # a point does not conflict with itself
    for p in range(128):
        if p < 64:
            adj_lo[p] &= ~(np.uint64(1) << np.uint64(p))
        else:
            adj_hi[p] &= ~(np.uint64(1) << np.uint64(p - 64))


@njit(cache=True)
def _csr_points(B_lo, B_hi, sel):
    cnt = np.zeros(129, np.int64)
    for t in range(sel.shape[0]):
        j = sel[t]
        for p in range(128):
            if _bit_test(B_lo[j], B_hi[j], p):
                cnt[p + 1] += 1
    ptr = np.cumsum(cnt)
    idx = np.empty(ptr[128], np.int64)
    pos = ptr[:128].copy()
    for t in range(sel.shape[0]):
        j = sel[t]
        for p in range(128):
            if _bit_test(B_lo[j], B_hi[j], p):
                idx[pos[p]] = j
                pos[p] += 1
    return ptr, idx


@njit(cache=True)
def survivor_bank_stats(E, B_lo, B_hi, w, Qid, target, res):
    """For each set E[i] (with u + 1_E = f): res[i,0] = min over bank items of
    wt(f+q) = w + k - 2|E cap B|;  res[i,1] = number of distinct Q (bank is
    sorted by Q) having an item with wt(f+q) == target."""
    n, k = E.shape
    for i in range(n):
        slo = np.uint64(0)
        shi = np.uint64(0)
        for t in range(k):
            p = E[i, t]
            if p < 64:
                slo |= np.uint64(1) << np.uint64(p)
            else:
                shi |= np.uint64(1) << np.uint64(p - 64)
        best = 1000
        cnt = 0
        lastQ = -1
        for j in range(B_lo.shape[0]):
            c = popcount64(B_lo[j] & slo) + popcount64(B_hi[j] & shi)
            v = w[j] + k - 2 * c
            if v < best:
                best = v
            if v == target and Qid[j] != lastQ:
                cnt += 1
                lastQ = Qid[j]
        res[i, 0] = best
        res[i, 1] = cnt


@njit(cache=True)
def survivors_full_check(E, B_lo, B_hi, bound, keep):
    """keep[i] = 1 iff survivor E[i] satisfies every bank inequality."""
    n, k = E.shape
    for i in range(n):
        slo = np.uint64(0)
        shi = np.uint64(0)
        for t in range(k):
            p = E[i, t]
            if p < 64:
                slo |= np.uint64(1) << np.uint64(p)
            else:
                shi |= np.uint64(1) << np.uint64(p - 64)
        ok = 1
        for j in range(B_lo.shape[0]):
            c = popcount64(B_lo[j] & slo) + popcount64(B_hi[j] & shi)
            if c > bound[j]:
                ok = 0
                break
        keep[i] = ok


# ---------------------------------------------------------------------------
# 5. Symmetry reduction with verified stabilizer generators
#
#    The catalogue lists generators of the affine stabilizer of each class
#    modulo RM(2,7).  Every generator used here is re-verified by truth-table
#    substitution: u(Ax+c) + u(x) must have degree <= 2.  ANY set of verified
#    elements generates a subgroup G of the stabilizer, and reducing modulo a
#    subgroup is always valid (it never needs the full stabilizer).
#    G maps the bank of error sets onto itself, and preserves the predicates
#    "sum zero", "sum nonzero" and "affinely independent"; hence the set of
#    surviving supports is G-invariant and it suffices to enumerate one
#    representative per orbit of prefixes:
#      level 1: E contains the minimum r of the first G-orbit O meeting E,
#               all points of E lie in orbits >= O;
#      level 2: same with G_r acting on the remaining allowed points;
#      level 3: same with G_{r,s}.
#    Orbits of G_r and G_{r,s} are read off the G-orbits of ordered pairs and
#    triples (union-find over 128^2 and 128^3 states).
# ---------------------------------------------------------------------------

_CONVENTION = {}


def verified_stabilizer(entry, t, n, convention):
    """Point permutations (np.int64[g, 2^n]) of the catalogue generators that
    verifiably fix t modulo RM(2,n)."""
    transpose, reverse, inverse = convention
    base = strip_low_degree(t, n, 2)
    perms = []
    for rows, tr in entry["gens"]:
        A, c = gen_to_affine(rows, tr, n, (transpose, reverse))
        Ai = invert_gf2_matrix(A, n)
        if Ai is None:
            continue
        if inverse:
            A, c = Ai, apply_linear(Ai, c)
        img = strip_low_degree(compose_truth_affine(t, n, A, c), n, 2)
        if img != base:
            continue
        perms.append([apply_linear(A, x) ^ c for x in range(1 << n)])
    if not perms:
        return np.zeros((0, 1 << n), np.int64)
    return np.array(perms, dtype=np.int64)


def verified_linear_symmetries(entry, t, n, convention):
    """Linear parts A (row masks) of the catalogue generators (A, c) that
    verifiably satisfy t(Ax+c) = t(x) + (degree <= 2).  For each such A the
    map R -> R o A + s is an automorphism of the profile of t."""
    transpose, reverse, inverse = convention
    base = strip_low_degree(t, n, 2)
    out = []
    for rows, tr in entry["gens"]:
        A, c = gen_to_affine(rows, tr, n, (transpose, reverse))
        Ai = invert_gf2_matrix(A, n)
        if Ai is None:
            continue
        if inverse:
            A, c = Ai, apply_linear(Ai, c)
        if strip_low_degree(compose_truth_affine(t, n, A, c), n, 2) == base:
            out.append(A)
    return out


@njit(cache=True)
def _uf_find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        nx = parent[x]
        parent[x] = r
        x = nx
    return r


@njit(cache=True)
def tuple_orbits(perms, arity):
    """Union-find labels of the G-orbits on ordered `arity`-tuples of points
    (states encoded base 128).  Label = minimal state of the orbit."""
    N = 128 ** arity
    parent = np.arange(N)
    for g in range(perms.shape[0]):
        pg = perms[g]
        for s in range(N):
            if arity == 1:
                t = pg[s]
            elif arity == 2:
                t = pg[s >> 7] * 128 + pg[s & 127]
            else:
                t = (pg[s >> 14] * 128 + pg[(s >> 7) & 127]) * 128 + pg[s & 127]
            a = _uf_find(parent, s)
            b = _uf_find(parent, t)
            if a != b:
                if a < b:
                    parent[b] = a
                else:
                    parent[a] = b
    for s in range(N):
        parent[s] = _uf_find(parent, s)
    return parent


def _mask_of(points):
    lo = 0
    hi = 0
    for p in points:
        if p < 64:
            lo |= 1 << p
        else:
            hi |= 1 << (p - 64)
    return np.uint64(lo), np.uint64(hi)


def normalization_cases(perms, depth, allowed=None):
    """List of (prefix points tuple, base points list) covering every k-set
    (k >= depth) up to the group generated by `perms`, as explained above.
    With no generators this is the single case ((), all points)."""
    pts = list(range(128)) if allowed is None else sorted(allowed)
    if perms.shape[0] == 0 or depth == 0:
        return [((), pts)], {"orbits1": None}
    labels = [None] + [tuple_orbits(perms, a) for a in range(1, depth + 1)]
    cases = []

    def rec(prefix, avail, level):
        if level > depth:
            cases.append((tuple(prefix), sorted(avail)))
            return
        lab = labels[level]
        key = 0
        for p in prefix:
            key = key * 128 + p
        groups = defaultdict(list)
        for x in avail:
            groups[int(lab[key * 128 + x])].append(x)
        orbs = sorted(groups.values(), key=lambda o: min(o))
        for i, orb in enumerate(orbs):
            r = min(orb)
            rest = set()
            for o in orbs[i:]:
                rest.update(o)
            rest.discard(r)
            rec(prefix + [r], rest, level + 1)

    rec([], set(pts), 1)
    return cases, {"orbits1": len(set(int(labels[1][x]) for x in pts))}


SUPPORT_MODES = {"any": 0, "zero_sum": 1, "nonzero_sum": 2, "affine_independent": 3}


def enumerate_supports(t, k, mode, D, wmax, bmax, perms, depth=3, P=None,
                       buf_rows=1 << 21, bank=None, cases=None):
    """All k-point supports E (one per normalisation case, i.e. at least one
    per orbit of the verified stabilizer subgroup) with d_2(t + 1_E) >= D,
    subject to the predicate `mode`.

    Requires wmax >= D + k - 2 so that the bank is exact.  A precomputed bank
    (with a larger weight limit) and precomputed cases may be passed in.
    Returns (E array, bank tuple, info dict)."""
    check(wmax >= D + k - 2, "bank too small for an exact test")
    md = SUPPORT_MODES[mode]
    if P is None:
        P = profile7(t)
    if bank is None:
        Blo, Bhi, w, Qid = quartic_bank(t, wmax, P)
    else:
        sel = bank[2] <= wmax
        Blo, Bhi, w, Qid = (x[sel] for x in bank)
    bound, ptr, idx, adj_lo, adj_hi, F0lo, F0hi, n0, n1, nm = build_dfs_bank(Blo, Bhi, w, k, D, bmax)
    if cases is None:
        cases, cinfo = normalization_cases(perms, depth if perms.shape[0] else 0)
    stats = np.zeros(1, np.int64)
    out = np.empty((buf_rows, k), np.int64)
    chunks = []
    splits = 0
    todo = list(cases)
    while todo:
        prefix, base = todo.pop()
        blo, bhi = _mask_of(base)
        n = support_dfs(k, md, Blo, Bhi, bound, ptr, idx, adj_lo, adj_hi, F0lo, F0hi,
                        blo, bhi, np.array(prefix, np.int64), out, buf_rows, stats)
        if n < 0:
            # too many raw survivors: split on the first free point
            splits += 1
            check(len(prefix) < k - 1, "cannot split further")
            for p in base:
                todo.append((tuple(prefix) + (p,), [x for x in base if x > p]))
            continue
        if n:
            chunks.append(out[:n].copy())
    raw = np.concatenate(chunks) if chunks else np.zeros((0, k), np.int64)
    useful = bound < k
    keep = np.zeros(len(raw), np.int64)
    if len(raw):
        survivors_full_check(raw, Blo[useful], Bhi[useful], bound[useful], keep)
    exact = raw[keep == 1]
    info = {"bank_items": int(len(w)), "bank_bound0": n0, "bank_bound1": n1,
            "bank_dfs_multi": nm, "cases": len(cases), "case_splits": splits,
            "dfs_nodes": int(stats[0]), "dfs_survivors": int(len(raw)),
            "exact_survivors": int(len(exact)), "generators": int(perms.shape[0])}
    return exact, (Blo, Bhi, w, Qid), info


# ---------------------------------------------------------------------------
# 6. Relative-position engine for pairs of profiles
#
#    For n-variable functions g, h with profiles C_g, C_h on the space Alt_n of
#    homogeneous quadratics, the restrictions of every concatenation
#    g || (h o A + quadratic + translation) have split distance
#        min_Q  C_g(Q) + C_h(Q o B + t),     B = A^{-1}, t in Alt_n
#    (input translations do not change a profile).  We decide whether some
#    (B, t) achieves split >= beta, by exhaustive search over B in GL(n,2)
#    with necessary star conditions, and over t with set conditions, finally
#    checking the full 2^dim profile arrays.  Q o B maps the basis quadratic
#    x_i x_j to (b_i.x)(b_j.x), b_i = row i of B.  With M = B^T and stars
#    W_l = {l ^ m}, Q o B maps W_l onto W_{M l}.
# ---------------------------------------------------------------------------

def wedge_vec(l, m, n):
    """Homogeneous part of (l.x)(m.x) as an Alt_n coefficient int."""
    v = 0
    for k, (i, j) in enumerate(PAIRS[n]):
        if (((l >> i) & (m >> j)) ^ ((l >> j) & (m >> i))) & 1:
            v |= 1 << k
    return v


def wedge_table(n):
    N = 1 << n
    W = np.zeros((N, N), np.int64)
    for l in range(N):
        for m in range(N):
            W[l, m] = wedge_vec(l, m, n)
    return W


_WEDGE = {}


def get_wedge(n):
    if n not in _WEDGE:
        _WEDGE[n] = wedge_table(n)
    return _WEDGE[n]


def star_bases(n):
    """For each nonzero l: echelon pivots (dict) of W_l = {l ^ m}."""
    W = get_wedge(n)
    return {l: gf2_basis([int(W[l, m]) for m in range(1 << n)]) for l in range(1, 1 << n)}


_STARB = {}


def star_basis_arrays(n):
    """Arrays (vec[l, r], piv[l, r], cnt[l]) of the echelon bases of the
    stars W_l, pivots in decreasing order."""
    if n not in _STARB:
        bases = star_bases(n)
        N = 1 << n
        vec = np.zeros((N, n), np.int64)
        piv = np.zeros((N, n), np.int64)
        cnt = np.zeros(N, np.int64)
        for l in range(1, N):
            items = sorted(bases[l].items(), reverse=True)
            cnt[l] = len(items)
            for r, (p, v) in enumerate(items):
                vec[l, r] = v
                piv[l, r] = p
        _STARB[n] = (vec, piv, cnt)
    return _STARB[n]


@njit(cache=True)
def _star_maxima_kernel(S, vec, piv, cnt, out):
    m = S.shape[0]
    red = np.empty(m, np.int64)
    for l in range(1, vec.shape[0]):
        for i in range(m):
            s = S[i]
            for r in range(cnt[l]):
                if (s >> piv[l, r]) & 1:
                    s ^= vec[l, r]
            red[i] = s
        red.sort()
        best = 0
        run = 0
        for i in range(m):
            if i > 0 and red[i] == red[i - 1]:
                run += 1
            else:
                run = 1
            if run > best:
                best = run
        out[l] = best


def star_maxima(S, n):
    """m_S(l) = max_t |S cap (t + W_l)| for every nonzero l (array index l)."""
    vec, piv, cnt = star_basis_arrays(n)
    out = np.zeros(1 << n, np.int64)
    S = np.asarray(S, dtype=np.int64)
    if len(S):
        _star_maxima_kernel(S, vec, piv, cnt, out)
    return out


def affine_dim(S):
    S = [int(s) for s in S]
    if not S:
        return -1
    return gf2_rank([s ^ S[0] for s in S[1:]])


def bipartite_perfect(adj):
    """adj[i] = list of allowed j.  True iff a perfect matching exists."""
    n = len(adj)
    match_r = [-1] * n

    def try_k(i, seen):
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_r[j] < 0 or try_k(match_r[j], seen):
                    match_r[j] = i
                    return True
        return False

    sys.setrecursionlimit(10000)
    for i in range(n):
        if not try_k(i, [False] * n):
            return False
    return True


def allowed_images(mLg, mHh, mLh, mHg, n):
    """allowed[l] = set of l' with mLg[l] <= mHh[l'] and mLh[l'] <= mHg[l]."""
    N = 1 << n
    al = {}
    for l in range(1, N):
        al[l] = [lp for lp in range(1, N) if mLg[l] <= mHh[lp] and mLh[lp] <= mHg[l]]
    return al


def bits_from_set(S, dim):
    b = np.zeros((1 << dim) // 64 if dim >= 6 else 1, np.uint64)
    for s in S:
        s = int(s)
        b[s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    return b


@njit(cache=True, inline="always")
def _inbits(bits, x):
    return (bits[x >> 6] >> np.uint64(x & 63)) & np.uint64(1)


@njit(cache=True)
def autocorrelation(S, dim):
    """AC[d] = #{(a,b) in S^2 : a ^ b = d}  (via Walsh-Hadamard)."""
    N = 1 << dim
    f = np.zeros(N, np.float64)
    for s in S:
        f[s] = 1.0
    h = 1
    while h < N:
        for b in range(0, N, 2 * h):
            for j in range(b, b + h):
                u = f[j]
                v = f[j + h]
                f[j] = u + v
                f[j + h] = u - v
        h *= 2
    for i in range(N):
        f[i] = f[i] * f[i]
    h = 1
    while h < N:
        for b in range(0, N, 2 * h):
            for j in range(b, b + h):
                u = f[j]
                v = f[j + h]
                f[j] = u + v
                f[j + h] = u - v
        h *= 2
    out = np.empty(N, np.int64)
    for i in range(N):
        out[i] = np.int64(np.round(f[i] / N))
    return out


@njit(cache=True)
def _lam_apply(tab, Q):
    return tab[0, Q & 127] ^ tab[1, (Q >> 7) & 127] ^ tab[2, (Q >> 14) & 127]


@njit(cache=True)
def _build_lam(cols, n, pi, pj, tab):
    """tab = lookup tables (3 x 128) of the linear map Q -> Q o B on Alt_n
    where row i of B is cols[i] (i.e. M e_i)."""
    dim = pi.shape[0]
    img = np.zeros(21, np.int64)
    for k in range(dim):
        a = cols[pi[k]]
        b = cols[pj[k]]
        v = 0
        for kk in range(dim):
            i = pi[kk]
            j = pj[kk]
            if (((a >> i) & (b >> j)) ^ ((a >> j) & (b >> i))) & 1:
                v |= 1 << kk
        img[k] = v
    for c in range(3):
        for x in range(128):
            v = 0
            for b in range(7):
                if (x >> b) & 1:
                    k = 7 * c + b
                    if k < dim:
                        v ^= img[k]
            tab[c, x] = v


@njit(cache=True, inline="always")
def _parity(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


@njit(cache=True)
def _bil(Yrows, a, b, n):
    """a^T Y b for an n x n GF(2) matrix given by row masks."""
    y = 0
    for i in range(n):
        if (a >> i) & 1:
            y ^= Yrows[i]
    return _parity(y & b)


@njit(cache=True)
def _reduce_vec(v, vec, piv, cnt):
    for r in range(cnt):
        if (v >> piv[r]) & 1:
            v ^= vec[r]
    return v


@njit(cache=True)
def _invert_cols(cols, n, inv_cols):
    rowsM = np.zeros(n, np.int64)
    for i in range(n):
        v = 0
        for j in range(n):
            if (cols[j] >> i) & 1:
                v |= 1 << j
        rowsM[i] = v
    A = rowsM.copy()
    Iv = np.zeros(n, np.int64)
    for i in range(n):
        Iv[i] = 1 << i
    for c in range(n):
        p = -1
        for r in range(c, n):
            if (A[r] >> c) & 1:
                p = r
                break
        tmp = A[c]
        A[c] = A[p]
        A[p] = tmp
        tmp = Iv[c]
        Iv[c] = Iv[p]
        Iv[p] = tmp
        for r in range(n):
            if r != c and (A[r] >> c) & 1:
                A[r] ^= A[c]
                Iv[r] ^= Iv[c]
    for j in range(n):
        v = 0
        for i in range(n):
            if (Iv[i] >> j) & 1:
                v |= 1 << i
        inv_cols[j] = v


@njit(cache=True)
def pair_search_kernel(n, allowed, pi, pj, Lg, Hh_list, Hh_bits, Lh, Hg_bits,
                       dg, ACh, dh, ACg, annY, annL_vec, annL_piv, annL_cnt,
                       annYr, annLr_vec, annLr_piv, annLr_cnt,
                       Cg, Ch, beta, budget, tri_label, stats, found):
    """Exhaustive search over M in GL(n,2) (columns c_k = M e_k, B = M^T).

    Inner-node conditions (all necessary for split >= beta):
      * stars:  M l in allowed[l] for every nonzero l;
      * annihilators: for every Y in annY (basis of Ann(D_{H_h})) the matrix
        M^T Y M lies in Ann(D_{L_g}); tested on the leading k x k blocks
        (annL_* = echelon bases of the truncations, per k).
    Leaf conditions: reverse annihilators (annYr = Ann(D_{H_g}) must map by
    M^{-T} . M^{-1} into Ann(D_{L_h})), difference counts, then every
    translation t with Lam(L_g)+t in H_h and Lam^{-1}(L_h+t) in H_g, and
    finally the full split min_Q Cg[Q] + Ch[Lam Q + t] against beta.
    Returns 1 if a counterexample was found (witness in `found`), 0 if the
    search finished, 2 if the inner-node budget was exhausted.
    stats: [inner nodes, complete maps, maps passing leaf tests,
            t candidates, t passing inclusions, full split checks]"""
    N = 1 << n
    cols = np.zeros(n, np.int64)
    img = np.zeros(N, np.int64)
    choice = np.zeros(n + 1, np.int64)
    tab = np.zeros((3, 128), np.int64)
    itab = np.zeros((3, 128), np.int64)
    inv_cols = np.zeros(n, np.int64)
    rY = annY.shape[0]
    T = np.zeros((n + 1, max(rY, 1)), np.int64)
    dim = pi.shape[0]
    k = 0
    choice[0] = 1
    while k >= 0:
        if k == n:
            stats[1] += 1
            if stats[0] + stats[1] > budget:
                return 2
            k -= 1
            _build_lam(cols, n, pi, pj, tab)
            ok = True
            for r in range(dg.shape[0]):
                if ACh[_lam_apply(tab, dg[r, 0])] < dg[r, 1]:
                    ok = False
                    break
            if not ok:
                continue
            _invert_cols(cols, n, inv_cols)
            # reverse annihilator condition
            for r in range(annYr.shape[0]):
                Z = 0
                for kk in range(dim):
                    if _bil(annYr[r], inv_cols[pi[kk]], inv_cols[pj[kk]], n):
                        Z |= 1 << kk
                if _reduce_vec(Z, annLr_vec, annLr_piv, annLr_cnt) != 0:
                    ok = False
                    break
            if not ok:
                continue
            _build_lam(inv_cols, n, pi, pj, itab)
            for r in range(dh.shape[0]):
                if ACg[_lam_apply(itab, dh[r, 0])] < dh[r, 1]:
                    ok = False
                    break
            if not ok:
                continue
            stats[2] += 1
            I0 = _lam_apply(tab, Lg[0])
            for hh in range(Hh_list.shape[0]):
                t = I0 ^ Hh_list[hh]
                stats[3] += 1
                good = True
                for r in range(1, Lg.shape[0]):
                    if not _inbits(Hh_bits, _lam_apply(tab, Lg[r]) ^ t):
                        good = False
                        break
                if good:
                    for r in range(Lh.shape[0]):
                        if not _inbits(Hg_bits, _lam_apply(itab, Lh[r] ^ t)):
                            good = False
                            break
                if good:
                    stats[4] += 1
                    stats[5] += 1
                    mn = 1000
                    for Q in range(1 << dim):
                        s = Cg[Q] + Ch[_lam_apply(tab, Q) ^ t]
                        if s < mn:
                            mn = s
                            if mn < beta:
                                break
                    if mn >= beta:
                        for i in range(n):
                            found[i] = cols[i]
                        found[n] = t
                        found[n + 1] = mn
                        return 1
            continue
        c = choice[k]
        if c >= N:
            k -= 1
            continue
        choice[k] = c + 1
        base = 1 << k
        ok = True
        for l in range(base):
            v = img[l] ^ c
            if v == 0:
                ok = False
                break
            if not (allowed[l | base, v >> 6] >> np.uint64(v & 63)) & np.uint64(1):
                ok = False
                break
        if not ok:
            continue
        if rY > 0:
            off = k * (k - 1) // 2
            for r in range(rY):
                y = 0
                for a in range(n):
                    if _parity(annY[r, a] & c):
                        y |= 1 << a
                v = T[k, r]
                for i in range(k):
                    if _parity(cols[i] & y):
                        v |= 1 << (off + i)
                if _reduce_vec(v, annL_vec[k], annL_piv[k], annL_cnt[k]) != 0:
                    ok = False
                    break
                T[k + 1, r] = v
            if not ok:
                continue
        if k == 2 and tri_label.shape[0] > 0:
            st = (cols[0] * 128 + cols[1]) * 128 + c
            if tri_label[st] != st:
                continue
        stats[0] += 1
        if stats[0] + stats[1] > budget:
            return 2
        for l in range(base):
            img[l | base] = img[l] ^ c
        cols[k] = c
        k += 1
        if k < n:
            choice[k] = 1
    return 0


def annihilator_basis(S, n):
    """Basis (list of Alt_n ints) of Ann(D) where D = direction space of the
    affine hull of S, for the coefficient pairing <Y, X> = sum_k Y_k X_k."""
    dim = n * (n - 1) // 2
    S = [int(s) for s in S]
    D = gf2_basis([s ^ S[0] for s in S[1:]])
    # solve <Y, d> = 0 for d in basis of D: nullspace of the matrix with rows d
    rows = list(D.values())
    # Gaussian elimination to reduced row echelon form, then nullspace
    piv_cols = []
    R = []
    for r in rows:
        for (pc, rr) in zip(piv_cols, R):
            if (r >> pc) & 1:
                r ^= rr
        if r:
            pc = r.bit_length() - 1
            for idx in range(len(R)):
                if (R[idx] >> pc) & 1:
                    R[idx] ^= r
            piv_cols.append(pc)
            R.append(r)
    free = [c for c in range(dim) if c not in piv_cols]
    basis = []
    for f in free:
        y = 1 << f
        for (pc, rr) in zip(piv_cols, R):
            if (rr >> f) & 1:
                y |= 1 << pc
        basis.append(y)
    for y in basis:
        for d in rows:
            check(bin(y & d).count("1") % 2 == 0, "annihilator computation")
    return basis


def _truncated_bases(vecs, n):
    """Echelon bases of the truncations of span(vecs) to the first k(k+1)/2
    coordinates (pairs inside columns 0..k), for k = 0..n-1."""
    vec = np.zeros((n, 21), np.int64)
    piv = np.zeros((n, 21), np.int64)
    cnt = np.zeros(n, np.int64)
    for k in range(n):
        m = (1 << (k * (k + 1) // 2)) - 1
        b = sorted(gf2_basis([v & m for v in vecs]).items(), reverse=True)
        cnt[k] = len(b)
        for r, (p, v) in enumerate(b):
            vec[k, r] = v
            piv[k, r] = p
    return vec, piv, cnt


def _rows_of_alt(Y, n):
    return np.array(alt_matrix_from_Q(Y, n), np.int64)


def alt_rank(Q, n):
    return gf2_rank(alt_matrix_from_Q(Q, n))


def annihilator_rank_obstruction(A_high, A_low, n, max_subspaces=2 * 10 ** 6):
    """Congruence Y -> M^T Y M maps span(A_high) injectively into span(A_low)
    preserving ranks.  Returns 'obstructed' if span(A_low) has no subspace of
    dimension dim span(A_high) with the same multiset of ranks, 'possible' if
    one exists, 'skipped' if there are too many subspaces to enumerate."""
    r = len(A_high)
    sdim = len(A_low)
    if r == 0:
        return "possible"
    if r > sdim:
        return "obstructed"
    target = sorted(alt_rank(_xor_select(A_high, a), n) for a in range(1, 1 << r))
    # number of r-dim subspaces of an sdim-dim space
    num = 1
    for i in range(r):
        num = num * ((1 << (sdim - i)) - 1) // ((1 << (i + 1)) - 1)
    if num > max_subspaces:
        return "skipped"
    rank_of = {}
    for a in range(1, 1 << sdim):
        rank_of[a] = alt_rank(_xor_select(A_low, a), n)
    # enumerate subspaces of F_2^sdim by reduced row echelon bases
    for pivots in itertools.combinations(range(sdim), r):
        free = [(i, c) for i in range(r) for c in range(sdim)
                if c > pivots[i] and c not in pivots]
        for fill in range(1 << len(free)):
            rows = [1 << pv for pv in pivots]
            for b, (i, c) in enumerate(free):
                if (fill >> b) & 1:
                    rows[i] |= 1 << c
            ranks = sorted(rank_of[_xor_select(rows, a)] for a in range(1, 1 << r))
            if ranks == target:
                return "possible"
    return "obstructed"


def pair_exclusion(Cg, Ch, n, beta, Lg_level, Hh_level, Lh_level, Hg_level,
                   diff_tests=24, budget=10 ** 11, h_symmetries=None):
    """Decide whether some relative position of g and h has split >= beta.

    L_g = {C_g <= Lg_level} must map into H_h = {C_h >= Hh_level} and
    L_h = {C_h <= Lh_level} must map back into H_g = {C_g >= Hg_level}; the
    callers choose levels with Lg_level + Hh_level == beta (and likewise),
    so these inclusions are implied by split >= beta.  Every relative
    position surviving the necessary tests is checked on the full profiles,
    so 'exhaustive_relative_position_search' is a complete exclusion."""
    dim = n * (n - 1) // 2
    Lg = np.nonzero(Cg <= Lg_level)[0]
    Hh = np.nonzero(Ch >= Hh_level)[0]
    Lh = np.nonzero(Ch <= Lh_level)[0]
    Hg = np.nonzero(Cg >= Hg_level)[0]
    res = {"L_g": int(len(Lg)), "H_h": int(len(Hh)), "L_h": int(len(Lh)), "H_g": int(len(Hg))}
    if len(Lg) == 0 or len(Lh) == 0:
        res["obstruction"] = "empty_low_set"
        return res
    if len(Lg) > len(Hh) or len(Lh) > len(Hg):
        res["obstruction"] = "cardinality"
        return res
    adg, adh = affine_dim(Lg), affine_dim(Hh)
    adl, adhg = affine_dim(Lh), affine_dim(Hg)
    res.update({"affdim_L_g": adg, "affdim_H_h": adh, "affdim_L_h": adl, "affdim_H_g": adhg})
    if adg > adh or adl > adhg:
        res["obstruction"] = "affine_hull_dimension"
        return res
    mLg, mHh, mLh, mHg = (star_maxima(S, n) for S in (Lg, Hh, Lh, Hg))
    al = allowed_images(mLg, mHh, mLh, mHg, n)
    if not bipartite_perfect([[x - 1 for x in al[l]] for l in range(1, 1 << n)]):
        res["obstruction"] = "star_maxima_matching"
        return res
    N = 1 << n
    allowed = np.zeros((N, 2), np.uint64)
    for l in range(1, N):
        for lp in al[l]:
            allowed[l, lp >> 6] |= np.uint64(1) << np.uint64(lp & 63)
    pi = np.array([i for (i, j) in PAIRS[n]], np.int64)
    pj = np.array([j for (i, j) in PAIRS[n]], np.int64)
    annH = annihilator_basis(Hh, n)
    annL = annihilator_basis(Lg, n)
    annHr = annihilator_basis(Hg, n)
    annLr = annihilator_basis(Lh, n)
    annY = np.array([_rows_of_alt(y, n) for y in annH], np.int64).reshape(-1, n)
    annL_vec, annL_piv, annL_cnt = _truncated_bases(annL, n)
    annYr = np.array([_rows_of_alt(y, n) for y in annHr], np.int64).reshape(-1, n)
    fb = sorted(gf2_basis(annLr).items(), reverse=True)
    annLr_vec = np.array([v for p, v in fb] + [0], np.int64)
    annLr_piv = np.array([p for p, v in fb] + [0], np.int64)
    annLr_cnt = len(fb)
    res.update({"ann_dims": [len(annH), len(annL), len(annHr), len(annLr)]})
    for (A1, A2, tag) in ((annH, annL, "forward"), (annHr, annLr, "reverse")):
        ob = annihilator_rank_obstruction(A1, A2, n)
        res["annihilator_rank_test_" + tag] = ob
        if ob == "obstructed":
            res["obstruction"] = "annihilator_rank_distribution"
            return res
    tri_label = np.zeros(0, np.int64)
    if h_symmetries is not None and len(h_symmetries):
        # columns of M transform by A^T for linear parts A of profile automorphisms of h
        # (points >= 2^n do not occur as columns; they are fixed so that the
        #  union-find never merges through them)
        permsT = np.array([[apply_linear(transpose_rows(A, n), x) if x < (1 << n) else x
                            for x in range(128)] for A in h_symmetries], np.int64)
        tri_label = tuple_orbits(permsT, 3)
        res["column_normalisation_generators"] = len(h_symmetries)
    ACh = autocorrelation(Hh.astype(np.int64), dim)
    ACg = autocorrelation(Hg.astype(np.int64), dim)
    ACLg = autocorrelation(Lg.astype(np.int64), dim)
    ACLh = autocorrelation(Lh.astype(np.int64), dim)

    def tests(AC):
        nz = np.nonzero(AC)[0]
        nz = nz[nz != 0]
        order = np.argsort(-AC[nz], kind="stable")
        sel = nz[order[:diff_tests]]
        return np.array([[int(d), int(AC[d])] for d in sel], np.int64).reshape(-1, 2)

    stats = np.zeros(6, np.int64)
    found = np.zeros(n + 2, np.int64)
    t0 = time.time()
    r = pair_search_kernel(n, allowed, pi, pj, Lg.astype(np.int64), Hh.astype(np.int64),
                           bits_from_set(Hh, dim), Lh.astype(np.int64), bits_from_set(Hg, dim),
                           tests(ACLg), ACh, tests(ACLh), ACg,
                           annY, annL_vec, annL_piv, annL_cnt,
                           annYr, annLr_vec, annLr_piv, annLr_cnt,
                           Cg.astype(np.int64), Ch.astype(np.int64), beta, budget, tri_label,
                           stats, found)
    res["search_seconds"] = round(time.time() - t0, 2)
    res["search_stats"] = {"inner_nodes": int(stats[0]), "complete_maps": int(stats[1]),
                           "maps_passing_leaf_tests": int(stats[2]), "t_candidates": int(stats[3]),
                           "t_passing_inclusions": int(stats[4]), "full_split_checks": int(stats[5])}
    if r == 1:
        res["obstruction"] = None
        res["COUNTEREXAMPLE"] = [int(x) for x in found]
    elif r == 2:
        res["obstruction"] = None
        res["INCOMPLETE"] = "search budget exhausted"
    else:
        res["obstruction"] = "exhaustive_relative_position_search"
    return res


def explicit_quadratic_certificate(t, n, Q, target):
    """Return a dict describing an explicit quadratic q (homogeneous part Q,
    linear part a, constant c) with wt(t+q) == target, verified by direct
    truth-table evaluation."""
    tq = t ^ quad_truth(Q, n)
    r = nearest_affine(tq, n, target)
    check(r is not None, "no affine completion attains the claimed distance")
    a, c = r
    q = quad_truth(Q, n) ^ linear_truth(a, n) ^ (((1 << (1 << n)) - 1) if c else 0)
    check(weight(t ^ q) == target, "explicit quadratic does not attain the distance")
    check(degree(q, n) <= 2, "explicit word is not quadratic")
    return {"Q": Q, "linear": a, "const": c, "distance": target}


# ===========================================================================
# STAGES
# ===========================================================================

STAGES = {}


def stage(name, heavy=False):
    def deco(fn):
        STAGES[name] = (fn, heavy)
        return fn
    return deco


def cert_status(ctx, name):
    p = ctx.cert_path(name)
    if not os.path.exists(p):
        return None
    return read_json(p).get("status")


def require(ctx, *names):
    for n in names:
        st = cert_status(ctx, n)
        check(st == "PASS", f"stage '{n}' must PASS first (status: {st}); run it before this stage")


# ---------------------------------------------------------------------------
# Stage: catalogues
# ---------------------------------------------------------------------------

def parse_cubic_forms(text):
    """Brier-Langevin list: lines 'NNN h=HEX +ijk+...' (digits 1..9 = x_1..x_9).
    Returns list of (index, set of frozenset triples over 0..8)."""
    out = []
    for raw in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+h=([0-9a-fA-F]+)\s*((?:\+\d{3})*)\s*$", raw)
        if not m:
            continue
        terms = set()
        for tok in re.findall(r"\+(\d{3})", m.group(3)):
            tri = frozenset(int(ch) - 1 for ch in tok)
            assert len(tri) == 3 and all(0 <= i < 9 for i in tri), raw
            terms ^= {tri}
        out.append((int(m.group(1)), terms))
    return out


@stage("catalogues")
def stage_catalogues(ctx, args):
    """Download the four public catalogues, record digests, parse them and
    check class counts."""
    t0 = time.time()
    res = {}
    inputs = {}
    entries, truths, info = load_quartic_catalogue(ctx)
    inputs["B-3-4-7.dat"] = info
    degs = Counter(degree(t, 7) for t in truths)
    res["quartic_classes"] = len(entries)
    res["quartic_degree_distribution"] = {str(k): v for k, v in sorted(degs.items())}
    conv = find_generator_convention(entries, 7, truths, 2)
    res["generator_convention"] = {"columns": conv[0], "reversed_bits": conv[1], "inverse": conv[2]}
    for name, n, expected in (("class-3-6.txt", 6, 205), ("class-2-6.txt", 6, 150357)):
        path, inf = fetch_catalogue(ctx, name)
        with open(path) as fh:
            ents = parse_class_file(fh.read(), n)
        check(len(ents) == expected, f"{name}: {len(ents)} classes, expected {expected}")
        inf["classes"] = len(ents)
        inputs[name] = inf
    path, inf = fetch_catalogue(ctx, "classification.txt")
    with open(path, errors="replace") as fh:
        cubics = parse_cubic_forms(fh.read())
    check(len(cubics) == 349 and [c[0] for c in cubics] == list(range(1, 350)),
          "cubic classification must list representatives 1..349")
    inf["classes"] = len(cubics)
    inputs["classification.txt"] = inf
    res["note"] = ("Completeness of these catalogues is an external mathematical input "
                   "(as in the paper); this stage checks only transport and class counts.")
    return write_certificate(ctx, "catalogues", "Catalogue inputs downloaded and parsed", "PASS",
                             res, inputs, started=t0)


# ---------------------------------------------------------------------------
# Stage: selftest  (two independent profile methods on random words)
# ---------------------------------------------------------------------------

@stage("selftest")
def stage_selftest(ctx, args):
    t0 = time.time()
    rng = np.random.default_rng(20261017)
    res = {}
    # ANF / truth table round trips
    for n in (6, 7, 8):
        for _ in range(20):
            t = int.from_bytes(rng.bytes(1 << (n - 3)), "little")
            check(mobius(mobius(t, n), n) == t, "Moebius transform is not an involution")
    # profile methods W and D agree; brute-force check of a few entries
    agree = 0
    for trial in range(int(args.get("samples", 6))):
        t = int.from_bytes(rng.bytes(16), "little")
        if trial % 2:
            t = truth_from_anf_string("abcde+abdfg+acefg+bcdef+abc+def", 7) ^ (1 << int(rng.integers(128)))
        P = profile7(t, "W")
        D = profile7(t, "D")
        check(np.array_equal(P, D), "profile methods W and D disagree")
        for Q in rng.integers(0, 1 << 21, 8):
            tq = t ^ quad_truth(int(Q), 7)
            brute = min(min(weight(tq ^ linear_truth(a, 7)), 128 - weight(tq ^ linear_truth(a, 7)))
                        for a in range(128))
            check(brute == P[Q], "profile entry disagrees with brute force")
        agree += 1
    res["profile_method_agreement"] = agree
    # parity facts used in the paper
    for _ in range(50):
        u = 0
        for m in range(128):
            if bin(m).count("1") <= 4 and rng.integers(2):
                u ^= monomial_truth(m, 7)
        q = 0
        for m in range(128):
            if bin(m).count("1") <= 2 and rng.integers(2):
                q ^= monomial_truth(m, 7)
        check((weight(u ^ q) - weight(u)) % 4 == 0, "wt(u+q) != wt(u) mod 4 for a quartic u")
    res["quartic_weight_mod4_samples"] = 50
    return write_certificate(ctx, "selftest", "Profile kernels agree with each other and with brute force",
                             "PASS", res, started=t0)


# ---------------------------------------------------------------------------
# Stage: witness  (lower bound 88)
# ---------------------------------------------------------------------------

WITNESS_ANF = "abc+adg+afh+agh+bdh+beg+ceh+def"
WITNESS_HEX = "01010101010101fe015432670154329801320e3d54675b975432680e01673da4"


@njit(cache=True, parallel=True)
def witness_walsh_kernel(w0lo, w0hi, w1lo, w1hi, QT6, C0, N0, C1, N1):
    """Method W with minimiser counts, for both 7-variable halves."""
    for q in prange(32768):
        qt = QT6[q]
        W0 = np.empty(64, np.int32)
        W1 = np.empty(64, np.int32)
        for half in range(2):
            lo = w0lo if half == 0 else w1lo
            hi = w0hi if half == 0 else w1hi
            walsh6(lo ^ qt, W0)
            walsh6(hi ^ qt, W1)
            for lam in range(64):
                m = 0
                cnt = 0
                for a in range(64):
                    s1 = abs(W0[a] + W1[a ^ lam])
                    s2 = abs(W0[a] - W1[a ^ lam])
                    if s1 > m:
                        m = s1
                        cnt = 1
                    elif s1 == m:
                        cnt += 1
                    if s2 > m:
                        m = s2
                        cnt = 1
                    elif s2 == m:
                        cnt += 1
                Q = q | (lam << 15)
                if half == 0:
                    C0[Q] = 64 - m // 2
                    N0[Q] = cnt
                else:
                    C1[Q] = 64 - m // 2
                    N1[Q] = cnt


@njit(cache=True, parallel=True)
def witness_direct_kernel(w0lo, w0hi, w1lo, w1hi, QT6, LIN6, L7lo, L7hi, C0, N0, C1, N1):
    """Method D: direct popcounts of each 128-bit half against all 256 affine
    functions; no Walsh transform and no 6-variable split."""
    for Q in prange(1 << 21):
        q = Q & 32767
        lam = Q >> 15
        qlo = QT6[q]
        qhi = QT6[q] ^ LIN6[lam]
        for half in range(2):
            glo = (w0lo if half == 0 else w1lo) ^ qlo
            ghi = (w0hi if half == 0 else w1hi) ^ qhi
            best = 1000
            cnt = 0
            for a in range(128):
                d = popcount64(glo ^ L7lo[a]) + popcount64(ghi ^ L7hi[a])
                for v in (d, 128 - d):
                    if v < best:
                        best = v
                        cnt = 1
                    elif v == best:
                        cnt += 1
            if half == 0:
                C0[Q] = best
                N0[Q] = cnt
            else:
                C1[Q] = best
                N1[Q] = cnt


@stage("witness")
def stage_witness(ctx, args):
    """d_2(w) = 88 for the cubic w = abc+adg+afh+agh+bdh+beg+ceh+def."""
    t0 = time.time()
    w = truth_from_anf_string(WITNESS_ANF, 8)
    check(degree(w, 8) == 3, "witness is not cubic")
    # decoder serialisation: f(0),f(1),... four per hex digit, MSB first
    bits = "".join(format(int(ch, 16), "04b") for ch in WITNESS_HEX)
    from_hex = sum(1 << x for x in range(256) if bits[x] == "1")
    check(from_hex == w, "hexadecimal witness string does not match the ANF")
    w0, w1 = w & ((1 << 128) - 1), w >> 128     # split on h = x_7
    args_ = [np.uint64(w0 & M64), np.uint64(w0 >> 64), np.uint64(w1 & M64), np.uint64(w1 >> 64)]
    results = {}
    for method in ("walsh_split6", "direct_popcount"):
        C0 = np.empty(1 << 21, np.int32)
        N0 = np.empty(1 << 21, np.int64)
        C1 = np.empty(1 << 21, np.int32)
        N1 = np.empty(1 << 21, np.int64)
        if method == "walsh_split6":
            witness_walsh_kernel(*args_, QT6, C0, N0, C1, N1)
        else:
            witness_direct_kernel(*args_, QT6, LIN6, LIN7_LO, LIN7_HI, C0, N0, C1, N1)
        S = C0 + C1
        d = int(S.min())
        arg = np.nonzero(S == d)[0]
        nearest = int((N0[arg] * N1[arg]).sum())
        hist = Counter((256 - 2 * S).tolist())
        results[method] = {"distance": d, "max_correlation": 256 - 2 * d,
                           "nearest_quadratic_count": nearest,
                           "minimising_homogeneous_parts": int(len(arg)),
                           "max_abs_correlation_histogram": {str(k): v for k, v in sorted(hist.items())}}
    a, b = results["walsh_split6"], results["direct_popcount"]
    check(a == b, "the two decoders disagree")
    check(a["distance"] == 88 and a["max_correlation"] == 80, "witness distance is not 88")
    # explicit nearest quadratic, verified on the full 8-variable truth table
    Qstar = int(arg[0])
    q7 = quad_truth(Qstar, 7)
    best = None
    for half, wh in ((0, w0), (1, w1)):
        target = int((C0 if half == 0 else C1)[Qstar])
        r = nearest_affine(wh ^ q7, 7, target)
        check(r is not None, "no affine completion")
        best = (best or []) + [r]
    (a0, c0), (a1, c1) = best
    full = ((1 << 128) - 1)
    h0 = q7 ^ linear_truth(a0, 7) ^ (full if c0 else 0)
    h1 = q7 ^ linear_truth(a1, 7) ^ (full if c1 else 0)
    q8 = h0 | (h1 << 128)
    check(degree(q8, 8) <= 2 and weight(w ^ q8) == 88, "explicit nearest quadratic fails")
    res = {"anf": WITNESS_ANF, "truth_hex": WITNESS_HEX, "decoders": results,
           "explicit_nearest_quadratic_anf": anf_string(mobius(q8, 8), 8),
           "paper_values": {"distance": 88, "max_correlation": 80, "nearest_count": 28672}}
    status = "PASS" if a["nearest_quadratic_count"] == 28672 else "PASS_WITH_DIFFERENT_COUNT"
    return write_certificate(ctx, "witness", "rho(2,8) >= 88: d_2(w) = 88 for the cubic witness",
                             status, res, started=t0)


# ---------------------------------------------------------------------------
# Stage: directions  (Section 7: good hyperplanes for eight-variable quintics)
# ---------------------------------------------------------------------------

def cubic_to_8var_types(cubics):
    """Forms of support dimension <= 8 among the nine-variable cubic forms,
    rewritten in eight variables (cubic part only, as sets of triples)."""
    types = []
    for idx, terms in cubics:
        # contraction matrices M_i[j][k] = [{i,j,k} in terms]
        def contraction(a):
            rows = [0] * 9
            for tri in terms:
                tl = tuple(tri)
                for i in tl:
                    if (a >> i) & 1:
                        j, k = [x for x in tl if x != i]
                        rows[j] ^= 1 << k
                        rows[k] ^= 1 << j
            return rows
        # radical = kernel of the linear map a -> contraction(a)
        images = []
        for i in range(9):
            rows = contraction(1 << i)
            images.append(Q_from_alt_matrix(rows, 9))
        radical = [a for a in range(1, 512)
                   if not _xor_select(images, a)]
        if not radical:
            continue
        r = min(radical)
        # basis with last vector r: complete r to a basis
        basis = []
        for e in range(9):
            cand = basis + [1 << e]
            if gf2_rank(cand + [r]) == len(cand) + 1:
                basis.append(1 << e)
            if len(basis) == 8:
                break
        cols = basis + [r]                  # A e_k = cols[k]
        rowsA = [sum(((cols[k] >> i) & 1) << k for k in range(9)) for i in range(9)]
        c = 0
        for tri in terms:
            m = sum(1 << i for i in tri)
            c ^= monomial_truth(m, 9)
        cA = compose_truth_affine(c, 9, rowsA, 0)
        a = mobius(cA, 9)
        new = set()
        for m in range(512):
            if (a >> m) & 1 and bin(m).count("1") == 3:
                check(not (m >> 8) & 1, "radical normalisation failed")
                new.add(frozenset(i for i in range(8) if (m >> i) & 1))
        types.append({"source_index": idx, "radical_dim": len(gf2_basis(radical)),
                      "triples": sorted(tuple(sorted(t)) for t in new)})
    return types


def _xor_select(vecs, a):
    v = 0
    i = 0
    while a:
        if a & 1:
            v ^= vecs[i]
        a >>= 1
        i += 1
    return v


def quintic_from_triples(triples, n=8):
    """Complement duality: cubic monomial x_S  ->  quintic monomial x_{S^c}."""
    full = (1 << n) - 1
    t = 0
    for tri in triples:
        m = full ^ sum(1 << i for i in tri)
        t ^= monomial_truth(m, n)
    return t


def restrict_hyperplane(t, n, a, c):
    """Truth table (n-1 variables) of t restricted to {x : a.x = c}, using the
    coordinates other than the lowest set bit p of a."""
    p = (a & -a).bit_length() - 1
    free = [i for i in range(n) if i != p]
    out = 0
    for y in range(1 << (n - 1)):
        x = 0
        for k, i in enumerate(free):
            if (y >> k) & 1:
                x |= 1 << i
        if (bin(a & x).count("1") & 1) != c:
            x |= 1 << p
        if (t >> x) & 1:
            out |= 1 << y
    return out


def leading_rank7(t):
    """Rank of the 7x7 alternating matrix of the degree-5 part of a
    seven-variable function (complementary pair of each quintic monomial)."""
    a = mobius(t, 7)
    rows = [0] * 7
    for m in range(128):
        if (a >> m) & 1 and bin(m).count("1") == 5:
            i, j = [k for k in range(7) if not (m >> k) & 1]
            rows[i] ^= 1 << j
            rows[j] ^= 1 << i
    return gf2_rank(rows)


def contraction_rank8(triples, a):
    rows = [0] * 8
    for tri in triples:
        for i in tri:
            if (a >> i) & 1:
                j, k = [x for x in tri if x != i]
                rows[j] ^= 1 << k
                rows[k] ^= 1 << j
    return gf2_rank(rows)


@stage("directions")
def stage_directions(ctx, args):
    """Section 7: (1) every quintic leading form has a normal of contraction
    rank 0 or 4; (2) for every vector v there is a normal in v-perp of rank
    0 or 4, or a normal outside v-perp of rank 0.  Also the contraction /
    restriction rank identity and the cubic <-> quintic orbit duality."""
    t0 = time.time()
    path, inf = fetch_catalogue(ctx, "classification.txt")
    with open(path, errors="replace") as fh:
        cubics = parse_cubic_forms(fh.read())
    check(len(cubics) == 349, "cubic catalogue must have 349 forms")
    types = cubic_to_8var_types(cubics)
    check(len(types) == 32, f"expected 32 eight-dimensional cubic types, found {len(types)}")
    res = {"types": len(types)}
    # (a) restriction rank == contraction rank on the 56 monomial basis forms
    basis_checks = 0
    for tri in itertools.combinations(range(8), 3):
        quint = quintic_from_triples([tri])
        for a in range(1, 256):
            r0 = leading_rank7(restrict_hyperplane(quint, 8, a, 0))
            r1 = leading_rank7(restrict_hyperplane(quint, 8, a, 1))
            check(r0 == r1 == contraction_rank8([tri], a), "rank identity fails on a basis form")
            basis_checks += 1
    res["basis_rank_identity_checks"] = basis_checks
    # (b) orbit duality  kappa(c) o A == kappa(c o A^{-T})  (mod degree <= 4)
    rng = np.random.default_rng(1)
    dual_ok = 0
    for _ in range(int(args.get("duality_samples", 40))):
        while True:
            rowsA = [int(rng.integers(1, 256)) for _ in range(8)]
            if gf2_rank(rowsA) == 8:
                break
        tri = [t for t in itertools.combinations(range(8), 3) if rng.integers(2)]
        c = 0
        for t3 in tri:
            c ^= monomial_truth(sum(1 << i for i in t3), 8)
        lhs = strip_low_degree(compose_truth_affine(quintic_from_triples(tri), 8, rowsA, 0), 8, 4)
        AinvT = transpose_rows(invert_gf2_matrix(rowsA, 8), 8)
        cB = mobius(compose_truth_affine(c, 8, AinvT, 0), 8)
        tri2 = [tuple(i for i in range(8) if (m >> i) & 1) for m in range(256)
                if (cB >> m) & 1 and bin(m).count("1") == 3]
        rhs = quintic_from_triples(tri2)
        check(lhs == rhs, "cubic/quintic duality identity fails")
        dual_ok += 1
    res["duality_identity_samples"] = dual_ok
    # (c) the two direction statements, with ranks computed from the actual
    #     hyperplane restrictions of the dual quintics
    alt1 = alt2 = 0
    type_records = []
    for ty in types:
        quint = quintic_from_triples(ty["triples"])
        rank = {}
        for a in range(1, 256):
            r0 = leading_rank7(restrict_hyperplane(quint, 8, a, 0))
            r1 = leading_rank7(restrict_hyperplane(quint, 8, a, 1))
            rc = contraction_rank8(ty["triples"], a)
            check(r0 == r1 == rc, "restriction rank != contraction rank on a type")
            rank[a] = r0
        good04 = [a for a in rank if rank[a] in (0, 4)]
        check(good04, "statement 1 fails for a type")
        witnesses = {}
        for v in range(1, 256):
            w1 = [a for a in good04 if bin(a & v).count("1") % 2 == 0]
            if w1:
                alt1 += 1
                witnesses[v] = ("perp_rank0or4", w1[0])
                continue
            w2 = [a for a in rank if rank[a] == 0 and bin(a & v).count("1") % 2 == 1]
            check(w2, f"statement 2 fails for type {ty['source_index']} and v={v}")
            alt2 += 1
            witnesses[v] = ("nonperp_rank0", w2[0])
        type_records.append({"source_index": ty["source_index"], "cubic_triples": ty["triples"],
                             "rank_counts": dict(Counter(rank.values())),
                             "statement1_witness": good04[0],
                             "statement2_witnesses": {str(v): w for v, w in witnesses.items()}})
    res.update({"type_vector_pairs": alt1 + alt2, "first_alternative": alt1,
                "second_alternative_only": alt2,
                "paper_values": {"pairs": 8160, "first": 8032, "second": 128}})
    write_json_gz(os.path.join(ctx.stage_dir("directions"), "types.json.gz"), type_records)
    return write_certificate(ctx, "directions", "Good hyperplane directions for 8-variable quintics",
                             "PASS", res, {"classification.txt": inf}, started=t0)


# ---------------------------------------------------------------------------
# Stage: algebra  (moment identities used in Sections 3.2 and 8.3)
# ---------------------------------------------------------------------------

def moments(t, n):
    """(weight parity, first moment v, second moment rows G) of a word."""
    v = 0
    rows = [0] * n
    w = 0
    x = 0
    tt = t
    while tt:
        if tt & 1:
            w ^= 1
            v ^= x
            for i in range(n):
                if (x >> i) & 1:
                    rows[i] ^= x
        tt >>= 1
        x += 1
    return w, v, rows


def restricted_rank(rows, v, n):
    """Rank of the bilinear form G restricted to v-perp."""
    perp = [a for a in range(1, 1 << n) if bin(a & v).count("1") % 2 == 0]
    B = list(gf2_basis(perp).values())
    M = [sum(_parity_py(B[i] & apply_linear(rows, B[j])) << j for j in range(len(B)))
         for i in range(len(B))]
    return gf2_rank(M)


def _parity_py(x):
    return bin(x).count("1") & 1


@stage("algebra")
def stage_algebra(ctx, args):
    """Exhaustive two-point moment reduction (Sec. 8.3), the ANF/moment
    identity, and spot checks of the quartic-distance formula (Sec. 3.2)."""
    t0 = time.time()
    rng = np.random.default_rng(7)
    res = {}
    # (1) ANF coefficients of degree n, n-1, n-2 are parity, first and second
    #     moments: coef[all]=w, coef[all-i]=w+v_i, coef[all-ij]=w+v_i+v_j+G_ij.
    n = 8
    full = (1 << n) - 1
    for _ in range(200):
        t = int.from_bytes(rng.bytes(32), "little")
        a = mobius(t, n)
        w, v, G = moments(t, n)
        check(((a >> full) & 1) == w, "top ANF coefficient != parity")
        for i in range(n):
            check(((a >> (full ^ (1 << i))) & 1) == (w ^ ((v >> i) & 1)), "degree-7 coefficient identity")
            for j in range(i + 1, n):
                c = (a >> (full ^ (1 << i) ^ (1 << j))) & 1
                check(c == (w ^ ((v >> i) & 1) ^ ((v >> j) & 1) ^ ((G[i] >> j) & 1)), "degree-6 identity")
    res["anf_moment_identity_samples"] = 200
    # (2) exhaustive: for every v != 0 and every symmetric G with diag G = v
    #     and G(v-perp) in <v>, some p gives pp^T + (p+v)(p+v)^T = G.
    total = 0
    for v in range(1, 256):
        perp = list(gf2_basis([a for a in range(1, 256) if not _parity_py(a & v)]).values())
        unknowns = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        eqs = []
        for a in perp:
            for y in perp:
                row = 0
                const = 0
                for k, (i, j) in enumerate(unknowns):
                    if (((y >> i) & (a >> j)) ^ ((y >> j) & (a >> i))) & 1:
                        row |= 1 << k
                for i in range(8):
                    const ^= (v >> i) & (y >> i) & (a >> i) & 1
                eqs.append((row, const))
        # solve
        piv = {}
        consistent = True
        for row, c in eqs:
            for p_, (r2, c2) in piv.items():
                if (row >> p_) & 1:
                    row ^= r2
                    c ^= c2
            if row == 0:
                if c:
                    consistent = False
                continue
            p_ = row.bit_length() - 1
            for q_ in list(piv):
                if (piv[q_][0] >> p_) & 1:
                    piv[q_] = (piv[q_][0] ^ row, piv[q_][1] ^ c)
            piv[p_] = (row, c)
        check(consistent, "moment equations inconsistent")
        free = [k for k in range(len(unknowns)) if k not in piv]
        pp_images = {}
        for p in range(256):
            q = p ^ v
            rows = [0] * 8
            for i in range(8):
                for j in range(8):
                    if (((p >> i) & (p >> j)) ^ ((q >> i) & (q >> j))) & 1:
                        rows[i] |= 1 << j
            pp_images.setdefault(tuple(rows), p)
        for fill in range(1 << len(free)):
            sol = 0
            for b, k in enumerate(free):
                if (fill >> b) & 1:
                    sol |= 1 << k
            for p_, (r2, c2) in piv.items():
                bit = c2 ^ _parity_py((r2 & ~(1 << p_)) & sol)
                if bit:
                    sol |= 1 << p_
            rows = [0] * 8
            for i in range(8):
                if (v >> i) & 1:
                    rows[i] |= 1 << i
            for k, (i, j) in enumerate(unknowns):
                if (sol >> k) & 1:
                    rows[i] |= 1 << j
                    rows[j] |= 1 << i
            # sanity: the conditions hold
            for a in perp:
                ga = apply_linear(rows, a)
                check(ga in (0, v), "solution violates G(v-perp) in <v>")
            check(tuple(rows) in pp_images, "no two-point word with the prescribed moments")
            total += 1
    res["two_point_moment_cases"] = total
    # (3) quartic-distance formula, spot checks in seven variables:
    #     (a) any even set E with first moment v != 0 has rank(G|v-perp) <= |E|-2;
    #     (b) an 8-point set inside a hyperplane has rank <= 4;
    #     (c) the paper's normal-form corrections realise rank r with r+2 points.
    n = 7
    cnt = Counter()
    for _ in range(int(args.get("formula_samples", 3000))):
        k = int(rng.choice([2, 4, 6, 8]))
        E = set()
        while len(E) < k:
            E.add(int(rng.integers(128)))
        t = sum(1 << p for p in E)
        w, v, G = moments(t, n)
        if v == 0:
            continue
        r = restricted_rank(G, v, n)
        check(r <= k - 2, "rank inequality rank(G|v-perp) <= |E|-2 fails")
        cnt[(k, r)] += 1
    res["rank_inequality_samples"] = {f"{k}:{r}": c for (k, r), c in sorted(cnt.items())}
    hp = 0
    for _ in range(500):
        a = int(rng.integers(1, 128))
        c = int(rng.integers(2))
        H = [x for x in range(128) if _parity_py(a & x) == c]
        E = set()
        while len(E) < 8:
            E.add(int(rng.choice(H)))
        t = sum(1 << p for p in E)
        w, v, G = moments(t, n)
        if v == 0:
            continue
        check(restricted_rank(G, v, n) <= 4, "8 points in a hyperplane with rank 6")
        hp += 1
    res["hyperplane_eight_point_samples"] = hp
    e = [1 << i for i in range(7)]
    for r in (0, 2, 4, 6):
        E = [0, e[6]] if r == 0 else [e[6]] + e[:r] + [sum(e[:r])]
        t = sum(1 << p for p in E)
        w, v, G = moments(t, n)
        check(w == 0 and v == e[6] and restricted_rank(G, v, n) == r and len(E) == r + 2,
              "normal-form correction does not realise its rank")
    res["normal_form_corrections"] = "ranks 0,2,4,6 realised by r+2 points"
    res["note"] = ("The quartic-distance formula itself is proved algebraically in the paper "
                   "(Sec. 3.2); these are consistency checks, as in the dossier.")
    return write_certificate(ctx, "algebra", "Moment identities and two-point reduction", "PASS",
                             res, started=t0)


# ---------------------------------------------------------------------------
# Stage: quartic_profiles  (all 68,443 classes; tables, labels, inequalities)
# ---------------------------------------------------------------------------

_QB7 = [monomial_truth((1 << i) | (1 << j), 7) for (i, j) in PAIRS[7]]
_LIN7 = [linear_truth(a, 7) for a in range(128)]
_ONES7 = (1 << 128) - 1


def quad7_int(Q):
    t = 0
    k = 0
    while Q:
        if Q & 1:
            t ^= _QB7[k]
        Q >>= 1
        k += 1
    return t


def explicit_nearest7(t, Q, target):
    tq = t ^ quad7_int(Q)
    for a in range(128):
        w = (tq ^ _LIN7[a]).bit_count()
        if w == target:
            return [Q, a, 0]
        if 128 - w == target:
            return [Q, a, 1]
    return None


def load_quartic_tables(ctx):
    d = ctx.stage_dir("quartic_profiles")
    p = os.path.join(d, "hist.npy")
    check(os.path.exists(p), "run stage quartic_profiles first")
    H = np.load(p)
    d2 = np.argmax(H > 0, axis=1)
    M = 64 - np.argmax(H[:, ::-1] > 0, axis=1)
    return H, d2, M


@stage("quartic_profiles", heavy=True)
def stage_quartic_profiles(ctx, args):
    """Full profiles of all 68,443 quartic classes: exact distances (with an
    explicit nearest quadratic each), histograms, and the inequalities
    (q36) N36 >= 64, (q38) M <= 54 & N38 >= 896, (q40) M <= 52."""
    t0 = time.time()
    entries, truths, info = load_quartic_catalogue(ctx)
    n = len(truths)
    d = ctx.stage_dir("quartic_profiles")
    lo, hi = truths_to_lohi(truths)
    chunk = int(args.get("chunk", 4096))
    for c0 in range(0, n, chunk):
        f = os.path.join(d, f"hist_{c0:06d}.npz")
        if os.path.exists(f):
            continue
        c1 = min(n, c0 + chunk)
        Hc = np.zeros((c1 - c0, 65), np.int64)
        Ac = np.zeros(c1 - c0, np.int64)
        batch_hist7(lo[c0:c1], hi[c0:c1], QT6, Hc, Ac)
        np.savez(f, H=Hc, A=Ac)
        ctx.log(f"quartic_profiles: classes {c0}..{c1 - 1} done ({time.time() - t0:.0f}s)")
    H = np.zeros((n, 65), np.int64)
    A = np.zeros(n, np.int64)
    for c0 in range(0, n, chunk):
        z = np.load(os.path.join(d, f"hist_{c0:06d}.npz"))
        H[c0:c0 + len(z["A"])] = z["H"]
        A[c0:c0 + len(z["A"])] = z["A"]
    check(bool((H.sum(axis=1) == (1 << 21)).all()), "a histogram does not sum to 2^21")
    np.save(os.path.join(d, "hist.npy"), H)
    d2 = np.argmax(H > 0, axis=1)
    M = 64 - np.argmax(H[:, ::-1] > 0, axis=1)
    # independent method D on a deterministic subset (all classes >= 36 and
    # every 61st class; --full-crosscheck for all)
    sub = sorted(set(np.nonzero(d2 >= 36)[0].tolist()) |
                 set(range(0, n, 1 if args.get("full_crosscheck") else 61)))
    Hd = np.zeros((len(sub), 65), np.int64)
    batch_profile7_direct(lo[sub], hi[sub], QT6, LIN6, Hd)
    check(np.array_equal(Hd, H[sub]), "method D disagrees with method W on a quartic class")
    # explicit nearest quadratic for every class
    nearest = []
    for i in range(n):
        r = explicit_nearest7(truths[i], int(A[i]), int(d2[i]))
        check(r is not None, f"class {i}: no explicit quadratic at the computed distance")
        nearest.append(r)
    write_json_gz(os.path.join(d, "nearest_quadratics.json.gz"),
                  {"format": "[Q (colex pair bits), linear mask, constant] per class", "rows": nearest})
    tiers = Counter(int(x) for x in d2)
    res = {"classes": n, "distance_distribution": {str(k): v for k, v in sorted(tiers.items())},
           "below_30": int((d2 < 30).sum())}
    for lev in (36, 38, 40):
        sel = np.nonzero(d2 == lev)[0]
        res[f"tier_{lev}"] = {"classes": int(len(sel)), "indices": sel.tolist(),
                              f"min_N{lev}": int(H[sel, lev].min()), "max_M": int(M[sel].max()),
                              "max_N_M": int(max(H[i, M[i]] for i in sel))}
    res["tier_38"]["min_N38"] = int(H[np.nonzero(d2 == 38)[0], 38].min())
    ineq = {
        "q36_N36_ge_64": res["tier_36"]["min_N36"] >= 64,
        "q38_M_le_54": res["tier_38"]["max_M"] <= 54,
        "q38_N38_ge_896": res["tier_38"]["min_N38"] >= 896,
        "q40_M_le_52": res["tier_40"]["max_M"] <= 52,
        "max_distance_le_40": int(d2.max()) <= 40,
    }
    res["inequalities"] = ineq
    res["method_D_crosscheck_classes"] = len(sub)
    res["hist_sha256"] = sha256_bytes(H.tobytes())
    res["paper_values"] = {"tier_36": 1636, "tier_38": 33, "tier_40": 4, "tier_34": 18158,
                           "tier_32": 29551}
    status = "PASS" if all(ineq.values()) else "FAIL"
    return write_certificate(ctx, "quartic_profiles", "Quartic catalogue tables (Sec. 3.1)", status,
                             res, {"B-3-4-7.dat": info}, started=t0)


# ---------------------------------------------------------------------------
# Stage: point_changes  (Lemma 5.1)
# ---------------------------------------------------------------------------

@njit(cache=True)
def nearest_error_sets(lo, hi, P, level, QT6, LIN6, out_lo, out_hi, out_meta):
    """All error sets supp(u+q) of the quadratics q with wt(u+q) == level."""
    n = 0
    for Q in range(1 << 21):
        if P[Q] != level:
            continue
        q = Q & 32767
        lam = Q >> 15
        glo = lo ^ QT6[q]
        ghi = hi ^ QT6[q] ^ LIN6[lam]
        for a in range(128):
            l_lo = LIN6[a & 63]
            l_hi = LIN6[a & 63]
            if a >= 64:
                l_hi = ~l_hi
            elo = glo ^ l_lo
            ehi = ghi ^ l_hi
            w = popcount64(elo) + popcount64(ehi)
            for c in range(2):
                if (w if c == 0 else 128 - w) == level:
                    if n >= out_lo.shape[0]:
                        return -1
                    out_lo[n] = elo if c == 0 else ~elo
                    out_hi[n] = ehi if c == 0 else ~ehi
                    out_meta[n, 0] = Q
                    out_meta[n, 1] = a
                    out_meta[n, 2] = c
                    n += 1
    return n


def greedy_point_cover(E_lo, E_hi):
    """Greedy cover of all 128 points by error sets; returns row indices or
    None if the union is not everything."""
    sets = [int(a) | (int(b) << 64) for a, b in zip(E_lo, E_hi)]
    full = _ONES7
    union = 0
    for s in sets:
        union |= s
    if union != full:
        return None
    covered = 0
    chosen = []
    while covered != full:
        best = max(range(len(sets)), key=lambda i: (sets[i] & ~covered).bit_count())
        chosen.append(best)
        covered |= sets[best]
    return chosen


@stage("point_changes")
def stage_point_changes(ctx, args):
    """Lemma 5.1: for quartics u, v and points p, q,
    d_2((u+delta_p) || (v+delta_q)) <= 90."""
    t0 = time.time()
    require(ctx, "quartic_profiles")
    entries, truths, info = load_quartic_catalogue(ctx)
    H, d2, M = load_quartic_tables(ctx)
    res = {}
    # distance 36: nearest error sets cover every point -> d_2(u+delta_p) = 35
    covers = {}
    max_sets = 0
    for i in np.nonzero(d2 == 36)[0]:
        t = truths[i]
        P = profile7(t)
        cap = 64 * int((P == 36).sum()) + 64
        out_lo = np.empty(cap, np.uint64)
        out_hi = np.empty(cap, np.uint64)
        meta = np.empty((cap, 3), np.int64)
        m = nearest_error_sets(np.uint64(t & M64), np.uint64(t >> 64), P, 36, QT6, LIN6,
                               out_lo, out_hi, meta)
        check(m > 0, "no nearest error sets")
        chosen = greedy_point_cover(out_lo[:m], out_hi[:m])
        check(chosen is not None, f"class {i}: nearest error sets do not cover all points")
        cov = []
        union = 0
        for r in chosen:
            Q, a, c = (int(x) for x in meta[r])
            q = quad7_int(Q) ^ _LIN7[a] ^ (_ONES7 if c else 0)
            e = t ^ q
            check(e.bit_count() == 36 and degree(q, 7) <= 2, "cover quadratic fails")
            union |= e
            cov.append([Q, a, c])
        check(union == _ONES7, "cover does not cover")
        covers[int(i)] = cov
        max_sets = max(max_sets, len(cov))
    res["distance36_classes"] = len(covers)
    res["distance36_cover_quadratics"] = sum(len(c) for c in covers.values())
    write_json_gz(os.path.join(ctx.stage_dir("point_changes"), "distance36_covers.json.gz"), covers)
    # distances 38 and 40: all 128 point changes profiled (two methods for a subset)
    for lev in (38, 40):
        cls = np.nonzero(d2 == lev)[0]
        words = [truths[i] ^ (1 << p) for i in cls for p in range(128)]
        lo, hi = truths_to_lohi(words)
        Hp = np.zeros((len(words), 65), np.int64)
        Ap = np.zeros(len(words), np.int64)
        batch_hist7(lo, hi, QT6, Hp, Ap)
        sub = list(range(0, len(words), 37))
        Hd = np.zeros((len(sub), 65), np.int64)
        batch_profile7_direct(lo[sub], hi[sub], QT6, LIN6, Hd)
        check(np.array_equal(Hd, Hp[sub]), "method D disagrees on point changes")
        dmin = np.argmax(Hp > 0, axis=1)
        dmax = 64 - np.argmax(Hp[:, ::-1] > 0, axis=1)
        lo_, hi_ = lev - 1, (55 if lev == 38 else 53)
        ok_min = bool((dmin == lo_).all())
        ok_max = bool((dmax <= hi_).all())
        ok_cnt = bool((Hp[:, lo_] > Hp[:, hi_]).all())
        res[f"distance{lev}"] = {"classes": int(len(cls)), "point_changes": len(words),
                                 "all_min_equal": lo_, "min_ok": ok_min, f"max_le_{hi_}": ok_max,
                                 f"N{lo_}_gt_N{hi_}": ok_cnt,
                                 f"min_N{lo_}_minus_N{hi_}": int((Hp[:, lo_] - Hp[:, hi_]).min()),
                                 "method_D_crosschecked": len(sub)}
        check(ok_min and ok_max and ok_cnt, f"point-change conditions fail at distance {lev}")
    res["paper_values"] = {"distance36_cover_quadratics": 30539, "distance38_cases": 33 * 128}
    res["logic"] = ("minima <= 35: 35+55=90; 37/39: 37+53=90; 37/37 and 39/39 by the strict "
                    "counting inequalities; an odd+odd split has even weight, so < 92 means <= 90.")
    return write_certificate(ctx, "point_changes", "Lemma 5.1 (two quartics, one point each)", "PASS",
                             res, started=t0)


# ---------------------------------------------------------------------------
# Stage: quartic_pairs  (Lemma 3.1, Section 4)
# ---------------------------------------------------------------------------

@stage("quartic_pairs")
def stage_quartic_pairs(ctx, args):
    """For quartics g, h in any relative position, d_2(g||h) <= 90, and <= 88
    if both have distance 36, 38 or 40."""
    t0 = time.time()
    require(ctx, "quartic_profiles")
    entries, truths, info = load_quartic_catalogue(ctx)
    conv = find_generator_convention(entries, 7, truths, 2)
    H, d2, M = load_quartic_tables(ctx)
    res = {"unequal_distances": {
        "min<=34": "34+56=90",
        "36/38": f"36+{int(M[d2 == 38].max())}",
        "36/40": f"36+{int(M[d2 == 40].max())}",
        "38/40": f"38+{int(M[d2 == 40].max())}"}}
    check(M[d2 == 38].max() <= 54 and M[d2 == 40].max() <= 52, "unequal-distance bounds fail")
    for lev in (36, 38, 40):
        cls = [int(i) for i in np.nonzero(d2 == lev)[0]]
        data = {}
        for i in cls:
            P = profile7(truths[i])
            L = np.nonzero(P == lev)[0]
            Hs = np.nonzero(P >= 92 - lev)[0]
            data[i] = (len(L), len(Hs), star_maxima(L, 7), star_maxima(Hs, 7))
        card = star = 0
        residual = []
        for a in range(len(cls)):
            for b in range(a, len(cls)):
                g, h = data[cls[a]], data[cls[b]]
                if g[0] > h[1] or h[0] > g[1]:
                    card += 1
                    continue
                al = allowed_images(g[2], h[3], h[2], g[3], 7)
                if not bipartite_perfect([[x - 1 for x in al[l]] for l in range(1, 128)]):
                    star += 1
                    continue
                residual.append((cls[a], cls[b]))
        rec = {"classes": len(cls), "unordered_pairs": len(cls) * (len(cls) + 1) // 2,
               "cardinality": card, "star_matching": star, "residual": []}
        for (a, b) in residual:
            Pa, Pb = profile7(truths[a]), profile7(truths[b])
            sym = verified_linear_symmetries(entries[b], truths[b], 7, conv)
            r = pair_exclusion(Pa, Pb, 7, 92, lev, 92 - lev, lev, 92 - lev,
                               budget=int(args.get("pair_budget", 10 ** 10)), h_symmetries=sym)
            r["pair"] = [a, b]
            rec["residual"].append(r)
            check("COUNTEREXAMPLE" not in r, f"pair {a},{b}: split distance >= 92 found")
            check(r.get("obstruction"), f"pair {a},{b}: not excluded ({r})")
        res[f"distance_{lev}"] = rec
    res["paper_values"] = {"36": {"cardinality": 1337840, "stars": 1220, "residual": 6},
                           "38": {"cardinality": 362, "stars": 199}, "40": {"star": 9, "residual": 1}}
    return write_certificate(ctx, "quartic_pairs", "Quartic pair theorem (Lemma 3.1)", "PASS", res,
                             started=t0)


# ---------------------------------------------------------------------------
# Six-variable helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def profile6_kernel(g, QT6, out):
    """out[q] = d(g+q, RM(1,6)) for all 2^15 homogeneous quadratics q."""
    W = np.empty(64, np.int32)
    for q in range(32768):
        walsh6(g ^ QT6[q], W)
        m = 0
        for a in range(64):
            v = abs(W[a])
            if v > m:
                m = v
        out[q] = 32 - m // 2


def profile6(g):
    out = np.empty(1 << 15, np.uint8)
    profile6_kernel(np.uint64(g), QT6, out)
    return out


def walsh_abs6(g):
    W = np.empty(64, np.int32)
    walsh6(np.uint64(g), W)
    return np.abs(W).astype(np.int64)


def load_six_catalogue(ctx, name):
    key = "six_" + name
    if key in ctx._cache:
        return ctx._cache[key]
    path, info = fetch_catalogue(ctx, name)
    with open(path) as fh:
        entries = parse_class_file(fh.read(), 6)
    truths = [truth_from_anf_string(e["anf"], 6) for e in entries]
    ctx._cache[key] = (entries, truths, info)
    return entries, truths, info


# ---------------------------------------------------------------------------
# Stage: degree6_bound  (Appendix A: degree-six seven-variable words have
#                        d_2 <= 38)
# ---------------------------------------------------------------------------

@stage("degree6_bound")
def stage_degree6_bound(ctx, args):
    """Split a degree-six word on a coordinate omitted by a sextic monomial:
    both halves are odd six-variable words.  For every pair of odd classes of
    RM(6,6)/RM(2,6) and every relative position the split distance is < 40."""
    t0 = time.time()
    entries, truths, info = load_six_catalogue(ctx, "class-3-6.txt")
    check(len(entries) == 205, "class-3-6 must have 205 classes")
    conv = find_generator_convention(entries, 6, truths, 2)
    odd = [i for i, t in enumerate(truths) if weight(t) % 2]
    res = {"classes": len(entries), "odd_classes": len(odd)}
    prof = {i: profile6(truths[i]) for i in odd}
    dmin = {i: int(prof[i].min()) for i in odd}
    dmax = {i: int(prof[i].max()) for i in odd}
    counts = Counter()
    hard = []
    for a in range(len(odd)):
        for b in range(a, len(odd)):
            i, j = odd[a], odd[b]
            if dmin[i] + dmax[j] <= 38 or dmin[j] + dmax[i] <= 38:
                counts["triangle"] += 1
                continue
            sym = verified_linear_symmetries(entries[j], truths[j], 6, conv)
            r = pair_exclusion(prof[i], prof[j], 6, 40, dmin[i], 40 - dmin[i], dmin[j], 40 - dmin[j],
                               budget=int(args.get("pair_budget", 10 ** 10)), h_symmetries=sym)
            check("COUNTEREXAMPLE" not in r, f"odd pair {i},{j}: split >= 40 found")
            check(r.get("obstruction"), f"odd pair {i},{j} not excluded: {r}")
            counts[r["obstruction"]] += 1
            if r["obstruction"] not in ("cardinality", "affine_hull_dimension"):
                r["pair"] = [i, j]
                r["anfs"] = [entries[i]["anf"], entries[j]["anf"]]
                hard.append(r)
    res["unordered_odd_pairs"] = len(odd) * (len(odd) + 1) // 2
    res["obstructions"] = dict(counts)
    res["pairs_beyond_cardinality_and_dimension"] = hard
    res["paper_values"] = {"odd_classes": 85, "pairs": 3655, "beyond_cardinality_and_dimension": 8}
    check(len(odd) == 85, "expected 85 odd classes")
    return write_certificate(ctx, "degree6_bound", "Degree-six seven-variable words have d_2 <= 38",
                             "PASS", res, {"class-3-6.txt": info}, started=t0)


# ---------------------------------------------------------------------------
# Stage: first56  (Section 3.3: degree-six words with d_2 = 36 and M = 56)
# ---------------------------------------------------------------------------

@njit(cache=True)
def first56_pair_kernel(Ai, Aj, Ci, Cj, orderQ, minCj, pi, pj, target, Uinv, b0,
                        out, max_out, stats):
    """All affine permutations pi'(y) = P' y + tau' of F_2^6 with
    Ai[y] + Aj[pi'(y)] <= 16 for every y, where Ai is the domain spectrum
    read in an adapted affine frame b = b0 + U y (so pi = pi' o frame^{-1},
    P = P' U^{-1}, tau = tau' + P b0).  For each, the split distance
    min_q Ci[q] + Cj[Lam_P q] is computed (early exit below target) and maps
    with split >= target are written to out (tau, P columns, split).
    stats: [nodes, compatible maps, maps reaching target]."""
    cols = np.zeros(6, np.int64)
    realP = np.zeros(6, np.int64)
    img = np.zeros(64, np.int64)
    choice = np.zeros(8, np.int64)
    tab = np.zeros((3, 128), np.int64)
    nout = 0
    for tau in range(64):
        if Ai[0] + Aj[tau] > 16:
            continue
        img[0] = tau
        k = 0
        choice[0] = 1
        while k >= 0:
            if k == 6:
                stats[1] += 1
                for m in range(6):
                    v = 0
                    for r in range(6):
                        if (Uinv[m] >> r) & 1:
                            v ^= cols[r]
                    realP[m] = v
                _build_lam(realP, 6, pi, pj, tab)
                mn = 1000
                for r in range(orderQ.shape[0]):
                    q = orderQ[r]
                    if Ci[q] + minCj >= mn:
                        break
                    s = Ci[q] + Cj[_lam_apply(tab, q)]
                    if s < mn:
                        mn = s
                        if mn < target:
                            break
                if mn >= target:
                    stats[2] += 1
                    if nout >= max_out:
                        return -1
                    tr = tau
                    for r in range(6):
                        if (b0 >> r) & 1:
                            tr ^= realP[r]
                    out[nout, 0] = tr
                    for i in range(6):
                        out[nout, 1 + i] = realP[i]
                    out[nout, 7] = mn
                    nout += 1
                k -= 1
                continue
            c = choice[k]
            if c >= 64:
                k -= 1
                continue
            choice[k] = c + 1
            base = 1 << k
            ok = True
            for b in range(base):
                # image of b | base is tau ^ P(b) ^ c = img[b] ^ c
                v = img[b] ^ c
                if Ai[b | base] + Aj[v] > 16:
                    ok = False
                    break
            if not ok:
                continue
            # linear independence of the new column
            dep = False
            for b in range(base):
                if (img[b] ^ tau) == c:
                    dep = True
                    break
            if dep:
                continue
            stats[0] += 1
            for b in range(base):
                img[b | base] = img[b] ^ c
            cols[k] = c
            k += 1
            if k < 6:
                choice[k] = 1
    return nout


def h_from_map(rep_j, tau, Pcols):
    """h'(x) = rep_j(P^{-T} x) + (P^{-1} tau).x  (six variables)."""
    Prows = [sum(((Pcols[k] >> i) & 1) << k for k in range(6)) for i in range(6)]
    Pinv = invert_gf2_matrix(Prows, 6)
    A = transpose_rows(Pinv, 6)            # P^{-T}
    lam = apply_linear(Pinv, tau)
    h = compose_truth_affine(rep_j, 6, A, 0)
    return h ^ linear_truth(lam, 6)


@njit(cache=True)
def adapted_frame_kernel(Ai, Aj):
    """Greedy affine frame b = b0 + U y of F_2^6 putting the most constrained
    points (fewest compatible images) first.  Returns (b0, U columns,
    score) where score estimates the log size of the first search levels."""
    cum = np.zeros(32, np.int64)
    for c in range(64):
        cum[Aj[c]] += 1
    for v in range(1, 32):
        cum[v] += cum[v - 1]
    lg = np.empty(64, np.float64)
    for b in range(64):
        lim = 16 - Ai[b]
        n = cum[lim] if lim >= 0 else 0
        lg[b] = np.log(max(1, n))
    b0 = 0
    for b in range(64):
        if lg[b] < lg[b0]:
            b0 = b
    U = np.zeros(6, np.int64)
    span = np.zeros(64, np.int64)
    inspan = np.zeros(64, np.bool_)
    inspan[0] = True
    ns = 1
    score = lg[b0]
    for k in range(6):
        best = 1e300
        bu = -1
        for u in range(1, 64):
            if inspan[u]:
                continue
            s = 0.0
            for t in range(ns):
                s += lg[b0 ^ u ^ span[t]]
            if s < best:
                best = s
                bu = u
        U[k] = bu
        for t in range(ns):
            span[ns + t] = span[t] ^ bu
            inspan[span[t] ^ bu] = True
        ns *= 2
        if k < 4:
            score += best
    return b0, U, score


def frame_data(Ai, Aj):
    b0, U, score = adapted_frame_kernel(Ai, Aj)
    U = [int(u) for u in U]
    rows = [sum(((U[k] >> i) & 1) << k for k in range(6)) for i in range(6)]
    inv = invert_gf2_matrix(rows, 6)
    Uinv = np.array([sum(((inv[i] >> m) & 1) << i for i in range(6)) for m in range(6)], np.int64)
    Ap = np.array([Ai[int(b0) ^ apply_linear(rows, y)] for y in range(64)], np.int64)
    return int(b0), Uinv, Ap, float(score)


_F56 = {}


def first56_prepare(ctx):
    """Candidate slices (odd six-variable classes with first-order distance
    >= 25) and their spectra/profiles, saved compactly for the workers."""
    d = ctx.stage_dir("first56")
    path = os.path.join(d, "slices.npz")
    if not os.path.exists(path):
        entries, truths, info = load_six_catalogue(ctx, "class-2-6.txt")
        cand = [i for i, t in enumerate(truths) if weight(t) % 2 and walsh_abs6(t).max() <= 14]
        spec = np.array([walsh_abs6(truths[i]) for i in cand], np.int64)
        prof = np.array([profile6(truths[i]) for i in cand], np.int16)
        order = np.array([np.argsort(p, kind="stable") for p in prof], np.int32)
        np.savez(path + ".tmp.npz", cand=np.array(cand, np.int64), spec=spec, prof=prof, order=order,
                 truths=np.array([truths[i] for i in cand], np.uint64))
        os.replace(path + ".tmp.npz", path)
    return path


def _first56_init(path):
    numba.set_num_threads(1)
    z = np.load(path)
    prof = z["prof"]
    _F56.update(cand=[int(x) for x in z["cand"]], spec=z["spec"], prof=prof, order=z["order"],
                dmin=[int(p.min()) for p in prof], dmax=[int(p.max()) for p in prof],
                pi=np.array([i for (i, j) in PAIRS[6]], np.int64),
                pj=np.array([j for (i, j) in PAIRS[6]], np.int64))


def first56_slice_work(a):
    """All pairs (slice a, slice b), b >= a.  Each unordered pair is searched
    in the orientation with the more constrained frame (both orientations
    enumerate the same configurations, pi <-> pi^{-1})."""
    F = _F56
    cand, spec, prof, dmin, dmax, order = (F[k] for k in ("cand", "spec", "prof", "dmin", "dmax", "order"))
    stats = np.zeros(3, np.int64)
    out = np.empty((1 << 14, 8), np.int64)
    tri = 0
    done = 0
    found = []
    for b in range(a, len(cand)):
        if dmin[a] + dmax[b] <= 34 or dmin[b] + dmax[a] <= 34:
            tri += 1
            continue
        done += 1
        f1 = frame_data(spec[a], spec[b])
        f2 = frame_data(spec[b], spec[a])
        x, y, (b0, Uinv, Ap, _) = (a, b, f1) if f1[3] <= f2[3] else (b, a, f2)
        m = first56_pair_kernel(Ap, spec[y], prof[x], prof[y], order[x], dmin[y], F["pi"], F["pj"],
                                36, Uinv, b0, out, out.shape[0], stats)
        check(m >= 0, "first56 output overflow")
        for r in range(m):
            found.append([cand[x], cand[y], int(out[r, 0]), [int(v) for v in out[r, 1:7]],
                          int(out[r, 7])])
    return {"slice": a, "triangle": tri, "pairs": done, "nodes": int(stats[0]),
            "maps": int(stats[1]), "maps_ge_36": int(stats[2]), "found": found}


@stage("first56")
def stage_first56(ctx, args):
    """Every degree-six f with d_2(f) = 36 and M(f) = 56 has N36 >= 60 and
    N56 <= 16 (paper: exactly N36 = 512, N56 = 16).

    After adding a quadratic, f has first-order distance 56; on a hyperplane
    split with a.v = 1 both halves are odd six-variable words g, h with
    |W_g(b)| + |W_h(b)| <= 16.  With g a catalogue representative and h in any
    relative position, |W_h| = |W_rep| o (affine permutation of frequencies);
    all such permutations are enumerated for every pair of slices."""
    t0 = time.time()
    entries, truths, info = load_six_catalogue(ctx, "class-2-6.txt")
    check(len(entries) == 150357, "class-2-6 must have 150,357 classes")
    res = {"classes": len(entries)}
    by_d1 = Counter()
    for t in truths:
        if weight(t) % 2:
            by_d1[32 - int(walsh_abs6(t).max()) // 2] += 1
    res["odd_classes_by_first_order_distance"] = {str(k): v for k, v in sorted(by_d1.items())}
    npz = first56_prepare(ctx)
    _first56_init(npz)
    cand = _F56["cand"]
    res["candidate_slices"] = len(cand)
    res["candidate_slices_by_d1"] = {str(k): v for k, v in sorted(
        Counter(32 - int(sp.max()) // 2 for sp in _F56["spec"]).items())}
    res["candidate_slice_max_profile"] = {str(k): v for k, v in sorted(Counter(_F56["dmax"]).items())}
    d = ctx.stage_dir("first56")
    part = os.path.join(d, "slices.jsonl")
    done = {}
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[r["slice"]] = r
                except json.JSONDecodeError:
                    pass
    todo = [a for a in range(len(cand)) if a not in done]
    workers = int(args.get("workers") or os.cpu_count() or 1)
    started = time.time()
    with open_append_clean(part) as fh:
        if workers <= 1 or not todo:
            it = (first56_slice_work(a) for a in todo)
            pool = None
        else:
            import multiprocessing as mp
            pool = mp.get_context("spawn").Pool(
                workers, initializer=_first56_init, initargs=(npz,))
            it = pool.imap_unordered(first56_slice_work, todo, chunksize=1)
        try:
            for n, r in enumerate(it, 1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[r["slice"]] = r
                if n % 50 == 0 or n == len(todo):
                    el = time.time() - started
                    ctx.log(f"first56: {len(done)}/{len(cand)} slices, {el/60:.1f} min, "
                            f"ETA {el/n*(len(todo)-n)/60:.1f} min")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    check(sorted(done) == list(range(len(cand))), "first56: missing slices")
    rs = list(done.values())
    res.update({"pairs_triangle_excluded": sum(r["triangle"] for r in rs),
                "pairs_enumerated": sum(r["pairs"] for r in rs),
                "search_nodes": sum(r["nodes"] for r in rs),
                "compatible_maps": sum(r["maps"] for r in rs),
                "maps_with_split_ge_36": sum(r["maps_ge_36"] for r in rs)})
    found = [x for r in rs for x in r["found"]]
    words = []
    for (i, j, tau, cols, mn) in found:
        h = h_from_map(truths[j], tau, cols)
        Ag, Ah = walsh_abs6(truths[i]), walsh_abs6(h)
        check(bool(((Ag + Ah) <= 16).all()), "reconstructed slice violates the Walsh inequality")
        f = truths[i] | (h << 64)
        P = profile7(f)
        hh = hist_from_profile(P)
        dd = int(P.min())
        check(dd == mn, "reconstructed word has a different split distance")
        check(degree(f, 7) == 6, "reconstructed word is not of degree six")
        check(int(P.max()) == 56, "reconstructed word does not have M = 56")
        words.append({"slices": [i, j], "tau": tau, "P_columns": cols, "truth": str(f),
                      "d2": dd, "M": int(P.max()), "N36": int(hh[36]), "N56": int(hh[56]),
                      "histogram": {str(k): int(v) for k, v in enumerate(hh) if v}})
    res["words"] = words
    check(all(w["d2"] <= 36 for w in words), "a degree-six word with d_2 >= 38 and M = 56 appeared")
    sel = [w for w in words if w["d2"] == 36]
    res["d2_36_M56_words"] = len(sel)
    if sel:
        res["min_N36"] = min(w["N36"] for w in sel)
        res["max_N56"] = max(w["N56"] for w in sel)
        check(res["min_N36"] >= 60 and res["max_N56"] <= 16, "six36 counting bounds fail")
    res["paper_values"] = {"slices_25": 983, "slices_27": 4, "compatible_maps": 248675,
                           "maps_ge_36": 2, "N36": 512, "N56": 16}
    return write_certificate(ctx, "first56", "Degree-six words with d_2=36 and M=56 (eq. six36)",
                             "PASS", res, {"class-2-6.txt": info}, started=t0)


# ---------------------------------------------------------------------------
# Heavy support stages: support6 (degree six, 6 points), affine8 (degree six,
# 8-point affine bases) and rank4 (rank-four quintics, 6 points of sum 0).
# One work unit = one quartic class.  Results are appended to shard files
# (JSON lines) and the stage certificate is produced by merging them.
# ---------------------------------------------------------------------------

SUPPORT_KINDS = {
    # kind: (k, mode, D, wmax, minimum quartic distance, bmax per tier)
    "support6": (6, "nonzero_sum", 38, 42, 32, {32: 3, 34: 3, 36: 3, 38: 4, 40: 4}),
    "affine8": (8, "affine_independent", 38, 44, 30, {30: 4, 32: 4, 34: 4, 36: 4, 38: 5, 40: 5}),
    "rank4": (6, "zero_sum", 36, 42, 30, {30: 3, 32: 3, 34: 2, 36: 3, 38: 4, 40: 4}),
}
SUPPORT_CLAIMS = {
    "support6": "Degree-six words u + 1_E, |E| = 6, around quartics with d_2 >= 32 (six38)",
    "affine8": "Degree-six words u + 1_E, E an affine basis, around quartics with d_2 >= 30 (six38)",
    "rank4": "Rank-four quintics: d_2 <= 38, (r438), and Lemma 3.3 (r436)",
}

_W = {}


def _worker_init(workdir):
    numba.set_num_threads(1)
    ctx = Ctx(workdir, quiet=True)
    entries, truths, info = load_quartic_catalogue(ctx)
    conv = find_generator_convention(entries, 7, truths, 2)
    H, d2, M = load_quartic_tables(ctx)
    _W.update(ctx=ctx, entries=entries, truths=truths, conv=conv, d2=d2)


def _word(t, E):
    for p in E:
        t ^= 1 << int(p)
    return t


def normalization_depth(stab_size):
    """Depth of the prefix normalisation (only affects speed: every depth is
    sound because the reduction uses verified group elements only)."""
    if not stab_size or stab_size <= 1:
        return 0
    if stab_size < 16:
        return 1
    if stab_size < 1000:
        return 2
    return 3


def supports_work(i, crosscheck=False):
    """All three support enumerations for one quartic class (shared profile,
    bank and symmetry cases).  Returns a JSON record."""
    t0 = time.time()
    entries, truths, conv, d2 = _W["entries"], _W["truths"], _W["conv"], _W["d2"]
    t = truths[i]
    du = int(d2[i])
    perms = verified_stabilizer(entries[i], t, 7, conv)
    P = profile7(t)
    check(int(P.min()) == du, "profile disagrees with the quartic table")
    bank = quartic_bank(t, 44, P)
    depth = normalization_depth(entries[i]["stab"]) if perms.shape[0] else 0
    cases, cinfo = normalization_cases(perms, depth)
    rec = {"class": int(i), "d2u": du, "generators": int(perms.shape[0]), "depth": depth,
           "cases": len(cases)}
    cand = None
    for kind in ("support6", "affine8", "rank4"):
        k, mode, D, wmax, dmin, bmaxes = SUPPORT_KINDS[kind]
        if du < dmin:
            continue
        tk = time.time()
        exact, (Blo, Bhi, w, Qid), info = enumerate_supports(
            t, k, mode, D, wmax, bmaxes[du], perms, depth, P, bank=bank, cases=cases)
        sub = dict(info)
        E_sorted = np.sort(exact, axis=1)
        if len(E_sorted):
            E_sorted = E_sorted[np.lexsort(E_sorted.T[::-1])]
        sub["survivors_sha256"] = sha256_bytes(np.ascontiguousarray(E_sorted).tobytes())
        stats = np.zeros((len(exact), 2), np.int64)
        if len(exact):
            survivor_bank_stats(exact, Blo, Bhi, w.astype(np.int64), Qid.astype(np.int64), D, stats)
        if kind == "rank4":
            sub["d2_36"] = int((stats[:, 0] == 36).sum()) if len(exact) else 0
            if cand is None:
                cand = candidate_lambda_masks(P, 50)
            m56, high = [], []
            for r, E in zip(stats, exact):
                f = _word(t, E)
                if r[0] == 36:
                    c = count_cost_at_least(np.uint64(f & M64), np.uint64(f >> 64), QT6, cand, 56)
                    if c:
                        m56.append([int(x) for x in E] + [int(r[1]), int(c)])
                else:
                    Pf = profile7(f)
                    h = hist_from_profile(Pf)
                    high.append([int(x) for x in E] +
                                [int(Pf.min()), int(Pf.max()), int(h[38]), int(h[54]), int(h[56])])
            sub["m56"] = m56
            sub["high"] = high
            if crosscheck and len(exact):
                f = _word(t, exact[0])
                Pf = profile7(f, "D")
                h = hist_from_profile(Pf)
                ok = int(Pf.min()) == int(stats[0, 0]) or (stats[0, 0] >= 38 and Pf.min() >= 38)
                if int(Pf.min()) == 36:
                    ok = ok and int(h[36]) == int(stats[0, 1])
                    c = count_cost_at_least(np.uint64(f & M64), np.uint64(f >> 64), QT6, cand, 56)
                    ok = ok and c == int(h[56])
                check(ok, f"class {i}: bank statistics disagree with an independent profile")
                sub["crosscheck"] = True
        else:
            found = []
            for r, E in zip(stats, exact):
                f = _word(t, E)
                check(degree(f, 7) == 6, "support word is not of degree six")
                Pf = profile7(f)
                h = hist_from_profile(Pf)
                row = [int(x) for x in E] + [int(Pf.min()), int(Pf.max()), int(h[38]), int(h[54])]
                if kind == "affine8":
                    w_, v_, G_ = moments(sum(1 << int(p) for p in E), 7)
                    row.append(restricted_rank(G_, v_, 7))
                found.append(row)
            sub["found"] = found
            if crosscheck and len(exact):
                f = _word(t, exact[0])
                check(int(profile7(f, "D").min()) == found[0][k], "independent profile disagrees")
                sub["crosscheck"] = True
        sub["seconds"] = round(time.time() - tk, 3)
        rec[kind] = sub
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def _supports_task(a):
    i, cc = a
    return supports_work(i, cc)


def supports_classes(ctx):
    H, d2, M = load_quartic_tables(ctx)
    return [int(i) for i in np.nonzero(d2 >= 30)[0]]


def read_shards(d):
    recs = {}
    for f in sorted(os.listdir(d)):
        if not f.startswith("shard_") or not f.endswith(".jsonl"):
            continue
        with open(os.path.join(d, f)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue      # torn last line after an interruption
                recs[r["class"]] = r
    return recs


@stage("supports", heavy=True)
def stage_supports(ctx, args):
    """Run the three support enumerations over every quartic class with
    d_2 >= 30 (sharded, resumable), then write the certificates support6,
    affine8 and rank4."""
    t0 = time.time()
    require(ctx, "quartic_profiles")
    d = ctx.stage_dir("supports")
    classes = supports_classes(ctx)
    shard, nsh = args.get("shard", (0, 1))
    mine = [c for idx, c in enumerate(classes) if idx % nsh == shard]
    if args.get("limit"):
        mine = mine[: int(args["limit"])]
    done = read_shards(d)
    todo = [c for c in mine if c not in done]
    ctx.log(f"supports: shard {shard}/{nsh}: {len(mine)} classes, {len(mine) - len(todo)} already done")
    H, d2, M = load_quartic_tables(ctx)
    todo.sort(key=lambda c: -int(d2[c]))          # the slowest tiers first
    out = os.path.join(d, f"shard_{shard:03d}_of_{nsh:03d}.jsonl")
    workers = int(args.get("workers") or os.cpu_count() or 1)
    tasks = [(c, (c % 16 == 0)) for c in todo]
    started = time.time()
    n_done = 0
    if tasks:
        with open_append_clean(out) as fh:
            if workers <= 1:
                _worker_init(ctx.workdir)
                it = (supports_work(*a) for a in tasks)
                pool = None
            else:
                import multiprocessing as mp
                pool = mp.get_context("spawn").Pool(
                    workers, initializer=_worker_init, initargs=(ctx.workdir,))
                it = pool.imap_unordered(_supports_task, tasks, chunksize=1)
            try:
                for rec in it:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_done += 1
                    if n_done % 250 == 0 or n_done == len(tasks):
                        el = time.time() - started
                        eta = el / n_done * (len(tasks) - n_done)
                        ctx.log(f"supports: {n_done}/{len(tasks)} classes, {el/60:.1f} min elapsed, "
                                f"ETA {eta/60:.1f} min (this shard)")
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
    certs = merge_supports(ctx, args, started=t0)
    st = [c["status"] for c in certs]
    agg = "PASS" if all(s == "PASS" for s in st) else ("FAIL" if "FAIL" in st else st[0])
    return {"status": agg}


def merge_supports(ctx, args, started=None):
    d = ctx.stage_dir("supports")
    classes = supports_classes(ctx)
    recs = read_shards(d)
    H, d2, M = load_quartic_tables(ctx)
    certs = []
    for kind in ("support6", "affine8", "rank4"):
        k, mode, D, wmax, dmin, bmaxes = SUPPORT_KINDS[kind]
        need = [c for c in classes if int(d2[c]) >= dmin]
        subs = {c: recs[c][kind] for c in need if c in recs and kind in recs[c]}
        missing = [c for c in need if c not in subs]
        res = {"kind": kind, "support_size": k, "predicate": mode, "target_distance": D,
               "bank_max_weight": wmax, "classes_required": len(need),
               "classes_done": len(subs),
               "dfs_nodes": sum(s["dfs_nodes"] for s in subs.values()),
               "exact_survivors": sum(s["exact_survivors"] for s in subs.values()),
               "cpu_seconds": round(sum(s["seconds"] for s in subs.values()), 1),
               "crosschecked_classes": sum(1 for s in subs.values() if s.get("crosscheck")),
               "by_quartic_distance": {}}
        for du in sorted(set(int(d2[c]) for c in subs)):
            cs = [c for c in subs if int(d2[c]) == du]
            res["by_quartic_distance"][str(du)] = {
                "classes": len(cs), "exact_survivors": sum(subs[c]["exact_survivors"] for c in cs),
                "cpu_seconds": round(sum(subs[c]["seconds"] for c in cs), 1)}
        ok = True
        if kind == "rank4":
            m56 = [[c] + x for c in subs for x in subs[c]["m56"]]
            high = [[c] + x for c in subs for x in subs[c]["high"]]
            d38 = [x for x in high if x[7] == 38]
            res.update({
                "d2_36_supports": sum(s["d2_36"] for s in subs.values()),
                "first56_positive_supports": len(m56),
                "min_N36_when_M56": min((x[7] for x in m56), default=None),
                "max_N56_when_M56": max((x[8] for x in m56), default=None),
                "d2_at_least_38_supports": len(high),
                "max_d2": max((x[7] for x in high), default=36),
                "d38": {"count": len(d38), "max_M": max((x[8] for x in d38), default=None),
                        "min_N38": min((x[9] for x in d38), default=None),
                        "max_N54": max((x[10] for x in d38), default=None)},
                "paper_values": {"retained_supports": 2887059, "first56_profiles": 5155,
                                 "min_N36": 60, "max_N56": 12,
                                 "d38": {"M": 54, "N38": 608, "N54": 576}}})
            ok = res["max_d2"] <= 38
            if m56:
                ok = ok and res["min_N36_when_M56"] >= 60 and res["max_N56_when_M56"] <= 12
            if d38:
                ok = ok and res["d38"]["max_M"] <= 54 and res["d38"]["min_N38"] >= 608 \
                    and res["d38"]["max_N54"] <= 576
            write_json_gz(os.path.join(d, "rank4_first56_positive.json.gz"),
                          {"columns": "class, E[6], N36, N56", "rows": m56})
            write_json_gz(os.path.join(d, "rank4_distance38_plus.json.gz"),
                          {"columns": "class, E[6], d2, M, N38, N54, N56", "rows": high})
        else:
            found = [[c] + x for c in subs for x in subs[c]["found"]]
            d38 = [x for x in found if x[k + 1] == 38]
            res.update({"degree_six_words_d2_ge_38": len(found),
                        "max_d2": max((x[k + 1] for x in found), default=None),
                        "d38": {"count": len(d38), "max_M": max((x[k + 2] for x in d38), default=None),
                                "min_N38": min((x[k + 3] for x in d38), default=None),
                                "max_N54": max((x[k + 4] for x in d38), default=None)}})
            if kind == "affine8":
                res["quartic_distance_8_words"] = sum(1 for x in found if x[-1] == 6)
            ok = all(x[k + 1] <= 38 for x in found)
            if d38:
                ok = ok and res["d38"]["max_M"] <= 54 and res["d38"]["min_N38"] >= 968 \
                    and res["d38"]["max_N54"] <= 8
            res["paper_values"] = {"six38": {"M": 54, "N38": 968, "N54": 8},
                                   "covering_list_sizes": [768, 1677, 824, 122, 58]}
            write_json_gz(os.path.join(d, f"{kind}_degree_six_words.json.gz"),
                          {"columns": f"class, E[{k}], d2, M, N38, N54" +
                                      (", rank" if kind == "affine8" else ""), "rows": found})
        if missing:
            status = "INCOMPLETE"
            res["missing_classes"] = len(missing)
        elif args.get("limit"):
            status = "PASS_SAMPLE_ONLY"
        else:
            status = "PASS" if ok else "FAIL"
        certs.append(write_certificate(ctx, kind, SUPPORT_CLAIMS[kind], status, res, started=started))
    return certs


# ---------------------------------------------------------------------------
# Stage: final  (assemble the proof of 88 <= rho(2,8) <= 91)
# ---------------------------------------------------------------------------

@stage("final")
def stage_final(ctx, args):
    t0 = time.time()
    names = ["catalogues", "selftest", "witness", "directions", "algebra", "quartic_profiles",
             "point_changes", "quartic_pairs", "degree6_bound", "first56", "support6", "affine8",
             "rank4"]
    certs = {}
    for n in names:
        p = ctx.cert_path(n)
        certs[n] = read_json(p) if os.path.exists(p) else None
    status_of = {n: (c["status"] if c else "MISSING") for n, c in certs.items()}
    R = {n: (c["results"] if c else None) for n, c in certs.items()}
    derived = {}
    problems = []
    try:
        q = R["quartic_profiles"]
        s6, a8, r4, f56 = R["support6"], R["affine8"], R["rank4"], R["first56"]
        # degree-six words with d_2 = 38: union of the 6-point and 8-point lists
        six38 = [x for x in (s6["d38"], a8["d38"]) if x["count"]]
        six38_M = max((x["max_M"] for x in six38), default=0)
        six38_N38 = min((x["min_N38"] for x in six38), default=10 ** 9)
        six38_N54 = max((x["max_N54"] for x in six38), default=0)
        r438 = r4["d38"]
        A38_minN38 = min([six38_N38] + ([r438["min_N38"]] if r438["count"] else []))
        A38_maxN54 = max([six38_N54] + ([r438["max_N54"]] if r438["count"] else []))
        A38_maxM = max([six38_M] + ([r438["max_M"]] if r438["count"] else []))
        six36_N36 = f56.get("min_N36", 10 ** 9)
        six36_N56 = f56.get("max_N56", 0)
        r436_N36 = r4["min_N36_when_M56"] if r4["min_N36_when_M56"] is not None else 10 ** 9
        r436_N56 = r4["max_N56_when_M56"] if r4["max_N56_when_M56"] is not None else 0
        derived = {
            "A_words_have_min_at_most_38": (R["degree6_bound"] is not None) and r4["max_d2"] <= 38,
            "Apair_36_38: max M over d2=38 members <= 54": A38_maxM <= 54,
            "Apair_38_38: min N38 > max N54 (both over d2=38 members)": A38_minN38 > A38_maxN54,
            "Apair_36_36: min N36 > max N56 (over d2=36, M=56 members)":
                min(six36_N36, r436_N36) > max(six36_N56, r436_N56),
            "mixed_36_36: quartic min N36 > degree-six max N56":
                q["tier_36"]["min_N36"] > six36_N56,
            "mixed_38_38: quartic min N38 > degree-six max N54": q["tier_38"]["min_N38"] > six38_N54,
            "mixed: quartic d2=40 has M <= 52": q["tier_40"]["max_M"] <= 52,
            "mixed: quartic d2=38 has M <= 54": q["tier_38"]["max_M"] <= 54,
            "mixed: degree-six d2=38 has M <= 54": six38_M <= 54,
        }
        derived["values"] = {"A38_min_N38": A38_minN38, "A38_max_N54": A38_maxN54,
                             "A38_max_M": A38_maxM, "six36_min_N36": six36_N36,
                             "six36_max_N56": six36_N56, "r436_min_N36": r436_N36,
                             "r436_max_N56": r436_N56}
    except Exception as e:  # some certificate missing
        problems.append(f"cannot evaluate derived inequalities: {e!r}")
    for n, st in status_of.items():
        if st != "PASS":
            problems.append(f"stage {n}: {st}")
    for k, v in derived.items():
        if k != "values" and not v:
            problems.append(f"derived inequality fails: {k}")
    chain = [
        "deg <= 4: Lemma 3.1 (quartic_pairs)",
        "deg 5: Section 7 statement 1 (directions) -> two quartics (quartic_pairs) or two "
        "rank-four quintics (Lemma 6.1: rank4 + derived)",
        "deg 6: both restrictions of degree six (Lemma 6.1: degree6_bound, support6, affine8, first56)",
        "deg 7: G a not in {0,v} -> Lemma 6.1; otherwise two-point reduction (algebra) and Section 7 "
        "statement 2 -> Lemma 6.2 (mixed) / 6.1 / 5.1 (point_changes)",
        "odd words: d_2(F) <= d_2(F + delta_p) + 1 <= 91",
        "lower bound: witness has distance 88 (witness)",
    ]
    external = [
        "completeness of the Gillot-Langevin RM(4,7)/RM(2,7) catalogue (68,443 classes)",
        "completeness of Langevin's RM(6,6)/RM(2,6) (205) and RM(6,6)/RM(1,6) (150,357) catalogues",
        "completeness of the Brier-Langevin cubic-form classification (349 forms in 9 variables)",
        "rho(1,7) = 56 (Mykkeltveit 1980; Hou 1996)",
        "rho(4,7) = 8 / the quartic-distance formula of Sec. 3.2 (algebraic)",
        "the algebraic reductions of the paper (split formula, moment and support arguments)",
    ]
    status = "PASS" if not problems else "INCOMPLETE"
    res = {"stage_status": status_of, "derived_inequalities": derived, "problems": problems,
           "logical_chain": chain, "external_inputs": external,
           "conclusion": ("88 <= rho(2,8) <= 91; even-weight words have distance <= 90"
                          if status == "PASS" else "not established by this run")}
    return write_certificate(ctx, "MASTER", "88 <= rho(2,8) <= 91 (conditional on external inputs)",
                             status, res, started=t0)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

ORDER = ["catalogues", "selftest", "witness", "directions", "algebra", "quartic_profiles",
         "point_changes", "quartic_pairs", "degree6_bound", "first56", "supports", "final"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Independent recomputation of the RM(2,8) covering-radius dossier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages (in order): " + ", ".join(ORDER) +
               "\n  'all' runs every stage; 'light' runs all but supports and final;"
               "\n  'merge' rebuilds the support6/affine8/rank4 certificates from the shard files.")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workdir", default=os.environ.get("RM28_WORKDIR", "rm28_work"))
    ap.add_argument("--threads", type=int, default=None, help="numba threads")
    ap.add_argument("--workers", type=int, default=None, help="processes for heavy stages")
    ap.add_argument("--shard", default="0/1", help="i/n: process the i-th of n shards")
    ap.add_argument("--limit", type=int, default=None, help="sample only (heavy stages)")
    ap.add_argument("--full-crosscheck", action="store_true")
    a = ap.parse_args(argv)
    ctx = Ctx(a.workdir, threads=a.threads)
    si, sn = (int(x) for x in a.shard.split("/"))
    args = {"workers": a.workers, "shard": (si, sn), "limit": a.limit,
            "full_crosscheck": a.full_crosscheck}
    todo = []
    for s in a.stages:
        if s == "all":
            todo += ORDER
        elif s == "light":
            todo += [x for x in ORDER if x not in ("supports", "final")]
        else:
            todo.append(s)
    ctx.log(f"rm28_verify {VERSION}; workdir {ctx.workdir}; numba threads {numba.get_num_threads()}")
    for s in todo:
        if s == "merge":
            merge_supports(ctx, args, started=time.time())
            continue
        if s not in STAGES:
            sys.exit(f"unknown stage {s}")
        ctx.log(f"=== stage {s} ===")
        try:
            cert = STAGES[s][0](ctx, args)
            ctx.log(f"=== {s}: {cert['status']} ===")
        except CheckFailure as e:
            write_certificate(ctx, s, "", "FAIL", {"error": str(e)})
            ctx.log(f"=== {s}: FAIL: {e} ===")
            raise


if __name__ == "__main__":
    main()
