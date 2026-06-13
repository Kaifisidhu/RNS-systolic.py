"""
rns_systolic.py -- Aryabhata's arithmetic inside the systolic array.

CLAIM UNDER TEST: the 32-bit partial-sum path (the widest carry chain in
design.sv) can be replaced by FOUR independent residue channels, moduli
{251,253,255,256}, every register <= 8 bits, NO carries crossing channels,
and the array computes the exact same matmul after ONE CRT decode at the
output port (where cim_adc_decoder sat in the earlier design).

This mirrors golden_model.py register-for-register: same input skew,
same output de-skew, same valid pipe, same two-phase commit, LATENCY=7.
Only the partial-sum registers changed: int32 -> 4x(<=8-bit residue).

Verified against (a) the proven golden model and (b) exact A@W,
with a per-cycle audit that no residue register ever exceeds 8 bits.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/claude/sphotonix")
sys.path.insert(0, "/home/claude/ancient")
from golden_model import run_matmul, N
from ancient_kernels import RNS

MODULI = (251, 253, 255, 256)
rns = RNS(MODULI)
LATENCY = 7


class RNSSystolicArray:
    """Exact mirror of golden SystolicArray; p-registers are residue channels."""
    def __init__(self, latency=LATENCY):
        self.latency = latency
        self.w_ch  = [np.zeros((N, N), dtype=np.int64) for _ in MODULI]
        self.a_reg = np.zeros((N, N), dtype=np.int64)          # int8 values
        self.p_ch  = [np.zeros((N, N), dtype=np.int64) for _ in MODULI]
        self.skew   = [np.zeros(k,     dtype=np.int64) for k in range(N)]
        self.deskew = [[np.zeros(N-1-c, dtype=np.int64) for c in range(N)]
                       for _ in MODULI]
        self.vpipe = np.zeros(latency, dtype=bool)
        self.max_reg_seen = 0          # per-cycle channel-width audit

    def load_weights(self, W):
        W = np.array(W, dtype=np.int64)
        for i, m in enumerate(MODULI):
            self.w_ch[i] = W % m       # weight residues, loaded once (stationary)

    def step(self, a_vec, in_valid):
        a_vec = np.asarray(a_vec, dtype=np.int64)
        a_into_col0 = np.array(
            [a_vec[0]] + [self.skew[k][-1] for k in range(1, N)], dtype=np.int64)

        next_a = np.empty_like(self.a_reg)
        next_p = [np.empty((N, N), dtype=np.int64) for _ in MODULI]
        for r in range(N):
            for c in range(N):
                a_in = a_into_col0[r] if c == 0 else self.a_reg[r, c-1]
                next_a[r, c] = a_in
                for i, m in enumerate(MODULI):
                    p_in = 0 if r == 0 else self.p_ch[i][r-1, c]
                    # the ONLY arithmetic in the PE: 8-bit modular MAC, carry-free
                    next_p[i][r, c] = (p_in + (a_in % m) * (self.w_ch[i][r, c])) % m

        next_skew = [self.skew[0]] + [
            np.concatenate(([a_vec[k]], self.skew[k][:-1])) for k in range(1, N)]

        next_deskew = []
        for i in range(len(MODULI)):
            pipes = []
            for c in range(N):
                pipe = self.deskew[i][c]
                pipes.append(np.concatenate(([self.p_ch[i][N-1, c]], pipe[:-1]))
                             if len(pipe) else pipe)
            next_deskew.append(pipes)

        next_v = np.concatenate(([in_valid], self.vpipe[:-1]))

        # commit (two-phase, like non-blocking assigns)
        self.a_reg, self.p_ch = next_a, next_p
        self.skew, self.deskew, self.vpipe = next_skew, next_deskew, next_v

        # per-cycle width audit: every residue register must fit in 8 bits
        cyc_max = max(int(ch.max()) for ch in self.p_ch)
        for i in range(len(MODULI)):
            for c in range(N):
                if len(self.deskew[i][c]):
                    cyc_max = max(cyc_max, int(self.deskew[i][c].max()))
        self.max_reg_seen = max(self.max_reg_seen, cyc_max)
        assert cyc_max < 256, "channel register exceeded 8 bits!"

        # output: residues at column bottoms -> ONE CRT decode at array edge
        out_res = []
        for i in range(len(MODULI)):
            out_res.append(np.array(
                [self.deskew[i][c][-1] if len(self.deskew[i][c])
                 else self.p_ch[i][N-1, c] for c in range(N)], dtype=np.int64))
        decoded = np.array([int(rns.decode(tuple(int(out_res[i][c])
                            for i in range(len(MODULI))))) for c in range(N)],
                           dtype=np.int64)
        return decoded, bool(self.vpipe[-1])


def rns_run_matmul(A, W):
    M = A.shape[0]
    arr = RNSSystolicArray()
    arr.load_weights(W)
    outs = []
    for t in range(M + LATENCY + 8):
        a_vec = A[t] if t < M else np.zeros(N, dtype=np.int64)
        out, v = arr.step(a_vec, t < M)
        if v: outs.append(out.copy())
    return np.array(outs, dtype=np.int64), arr.max_reg_seen


def main():
    rng = np.random.default_rng(7)
    fails, trials, worst_reg, max_elem = 0, 300, 0, 0
    for _ in range(trials):
        M = int(rng.integers(1, 13))
        A = rng.integers(-128, 128, size=(M, N)).astype(np.int64)
        W = rng.integers(-128, 128, size=(N, N)).astype(np.int64)
        ref = A @ W
        g = run_matmul(A, W, LATENCY)            # proven golden, int32 path
        r, reg = rns_run_matmul(A, W)            # carry-free residue path
        if not np.array_equal(g, ref): fails += 1
        if not np.array_equal(r, ref): fails += 1
        if not np.array_equal(r, g):   fails += 1
        worst_reg = max(worst_reg, reg)
        max_elem = max(max_elem, int(np.abs(ref).max()))
    ok = fails == 0
    print(f"[{'PASS' if ok else 'FAIL'}] RNS array == golden array == A@W, "
          f"{trials} random trials (M=1..12), 0 tolerated mismatches: {fails} fails")
    print(f"[PASS] per-cycle width audit: largest residue register ever = "
          f"{worst_reg} < 256  ->  every p-register is 8 bits")
    print(f"[INFO] largest |output| seen = {max_elem}  (RNS exact up to "
          f"{rns.M//2:,}; with our bound, dots up to L=126,509 terms)")
    print()
    print("Honest cost ledger (per PE, per cycle):")
    print("  binary path : 8x8 mult + 32-bit ripple add  (carry depth ~32)")
    print("  RNS path    : 4x parallel (8x8 modular mult + 8-bit modular add)")
    print("  wins: carry DEPTH (clock speed lever)   loses: AREA (~4x multipliers)")
    print("  decode = one CRT block at array edge, off the per-cycle hot path")
    print("  Same tradeoff as the published RNS-TPU patents. A lever, not magic.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
