#include <metal_stdlib>
using namespace metal;

// Metal heads (docs/spec/01-network.md §4, T24). Reads bits straight out of the *packed* last
// gate layer (`[G,S,S,C]` uint32, `PackBits` layout) so `MetalPackedBackend` never has to unpack
// on the CPU between gates and heads. Compiled with fast-math OFF (`MTLCompileOptions.
// fastMathEnabled = false`), FP32 throughout, matching `Heads.evaluate`'s contract
// (docs/spec/05-validation.md §3 tolerance, not bit-exactness -- this is still a reassociated sum
// order vs. the CPU explicit loop).
//
// `ModelManifest.localHidden`/`globalHidden` are fixed Swift constants (64/128), independent of
// channel count or headVersion, so they are compiled in here rather than passed as dims -- that
// keeps every per-thread accumulator array a fixed, small, stack-sized array instead of a
// variable-length one (Metal has no VLAs).
#define LOCAL_HIDDEN 64
#define GLOBAL_HIDDEN 128
// headVersion 3 (docs/spec/01-network.md §4): a 3x3 grid of regions covering the board; zreg
// flattens to REGION_COUNT*LOCAL_HIDDEN = 9*64 = 576 values, region (i,j) at i*REGION_GRID+j.
#define REGION_GRID 3
#define REGION_COUNT 9
#define REGION_HIDDEN 576

// Step 1/3: m_c = mean_xy(h_c), v_c = max_xy(h_c) per (batch, channel), read directly from the
// packed bit tensor. One thread per (c, b).
//   dims = [S, C, B]
kernel void heads_reduce(
    device const uint* h    [[buffer(0)]],
    device float* mOut      [[buffer(1)]],
    device float* vOut      [[buffer(2)]],
    device const int* dims  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int S = dims[0];
    const int C = dims[1];
    const int B = dims[2];
    const int c = int(gid.x);
    const int b = int(gid.y);
    if (c >= C || b >= B) return;

    const int group = b / 32;
    const uint k = uint(b % 32);
    float sum = 0.0f;
    float mx = 0.0f;
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            const int idx = ((group * S + y) * S + x) * C + c;
            const float bit = float((h[idx] >> k) & 1u);
            sum += bit;
            mx = max(mx, bit);
        }
    }
    mOut[b * C + c] = sum / float(S * S);
    vOut[b * C + c] = mx;
}

// Step 2/3: local projection per point.
//   u_xy = concat(h_xy, m, v, global)            // 3C+4
//   z_xy = ReLU(u_xy @ Wlocal + blocal)           // 64
//   policy_xy = z_xy @ Wpolicy + bpolicy
//   ownership_xy = tanh(z_xy @ Wowner + bowner)
// `u_xy` is never materialised: each input segment (h/m/v/global) accumulates directly into the
// 64-wide `z` accumulator, avoiding a per-point `3C+4`-length temporary. One thread per (x,y,b);
// writes `zBuf` too (consumed by `heads_global`'s zbar/ownMean reduction, headVersion 2 only).
//   dims = [S, C, B]
kernel void heads_local(
    device const uint* h           [[buffer(0)]],
    device const float* mIn        [[buffer(1)]],
    device const float* vIn        [[buffer(2)]],
    device const float* globalIn   [[buffer(3)]],
    device const float* Wlocal     [[buffer(4)]],
    device const float* blocal     [[buffer(5)]],
    device const float* Wpolicy    [[buffer(6)]],
    device const float* bpolicy    [[buffer(7)]],
    device const float* Wowner     [[buffer(8)]],
    device const float* bowner     [[buffer(9)]],
    device float* policyLogits     [[buffer(10)]], // [B,P+1]; this kernel only writes indices < P
    device float* ownership        [[buffer(11)]], // [B,P]
    device float* zBuf             [[buffer(12)]], // [B,P,LOCAL_HIDDEN]
    device const int* dims         [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int S = dims[0];
    const int C = dims[1];
    const int B = dims[2];
    const int x = int(gid.x);
    const int y = int(gid.y);
    const int b = int(gid.z);
    if (x >= S || y >= S || b >= B) return;
    const int P = S * S;
    const int p = y * S + x;
    const int group = b / 32;
    const uint k = uint(b % 32);

    float z[LOCAL_HIDDEN];
    for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] = blocal[j];

    const int pointBase = ((group * S + y) * S + x) * C;
    for (int c = 0; c < C; c++) {
        const float bit = float((h[pointBase + c] >> k) & 1u);
        const int row = c * LOCAL_HIDDEN;
        for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] += bit * Wlocal[row + j];
    }
    for (int c = 0; c < C; c++) {
        const float mval = mIn[b * C + c];
        const int row = (C + c) * LOCAL_HIDDEN;
        for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] += mval * Wlocal[row + j];
    }
    for (int c = 0; c < C; c++) {
        const float vval = vIn[b * C + c];
        const int row = (2 * C + c) * LOCAL_HIDDEN;
        for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] += vval * Wlocal[row + j];
    }
    for (int i = 0; i < 4; i++) {
        const float gval = globalIn[b * 4 + i];
        const int row = (3 * C + i) * LOCAL_HIDDEN;
        for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] += gval * Wlocal[row + j];
    }
    for (int j = 0; j < LOCAL_HIDDEN; j++) z[j] = max(0.0f, z[j]);

    float pol = bpolicy[0];
    float own = bowner[0];
    for (int j = 0; j < LOCAL_HIDDEN; j++) {
        pol += z[j] * Wpolicy[j];
        own += z[j] * Wowner[j];
    }
    policyLogits[b * (P + 1) + p] = pol;
    ownership[b * P + p] = tanh(own);
    const int zBase = (b * P + p) * LOCAL_HIDDEN;
    for (int j = 0; j < LOCAL_HIDDEN; j++) zBuf[zBase + j] = z[j];
}

// Step 3/3: global projection, one thread per batch sample. headVersion 2/3 first reduce `zbar =
// mean_xy(z_xy)` and `ownMean = mean_xy(ownership)` from `heads_local`'s per-point outputs (B is
// small -- at most a few hundred -- so this per-thread P-length reduction is cheap relative to
// the C*128 projection below; it avoids a fourth kernel/dispatch). headVersion 3 additionally
// reduces `zreg` (the mean of z_xy over each cell of a 3x3 grid of regions, docs/spec/01-network.md
// §4) in the SAME single pass over P, using integer region boundaries derived from S (matching
// `ichigo_train.model.region_bounds` / `Heads.regionBounds`: bounds[i] = (i*S)/REGION_GRID).
//   u_global = concat(m, v, global)                          // 2C+4   (headVersion 1)
//   u_global = concat(m, v, zbar, ownMean, global)            // 2C+69  (headVersion 2)
//   u_global = concat(m, v, zbar, zreg, ownMean, global)      // 2C+645 (headVersion 3)
//   z_global = ReLU(u_global @ Wglobal + bglobal)             // 128
//   passLogit = z_global @ Wpass + bpass
//   wdlLogits = z_global @ Wwdl + bwdl                        // 3
//   scoreMean = z_global @ Wscore + bscore
//   dims = [S, C, B, headVersion]
kernel void heads_global(
    device const float* mIn        [[buffer(0)]],
    device const float* vIn        [[buffer(1)]],
    device const float* zBuf       [[buffer(2)]], // [B,P,LOCAL_HIDDEN], headVersion 2/3 only
    device const float* ownership  [[buffer(3)]], // [B,P], headVersion 2/3 only
    device const float* globalIn   [[buffer(4)]],
    device const float* Wglobal    [[buffer(5)]],
    device const float* bglobal    [[buffer(6)]],
    device const float* Wpass      [[buffer(7)]],
    device const float* bpass      [[buffer(8)]],
    device const float* Wwdl       [[buffer(9)]],
    device const float* bwdl       [[buffer(10)]],
    device const float* Wscore     [[buffer(11)]],
    device const float* bscore     [[buffer(12)]],
    device float* policyLogits     [[buffer(13)]], // [B,P+1]; this kernel only writes index P (pass)
    device float* wdlLogits        [[buffer(14)]], // [B,3]
    device float* scoreMean        [[buffer(15)]], // [B]
    device const int* dims         [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    const int S = dims[0];
    const int C = dims[1];
    const int B = dims[2];
    const int headVersion = dims[3];
    const int b = int(gid);
    if (b >= B) return;
    const int P = S * S;

    float zbar[LOCAL_HIDDEN];
    float ownMean = 0.0f;
    float zreg[REGION_HIDDEN];
    if (headVersion == 2 || headVersion == 3) {
        for (int j = 0; j < LOCAL_HIDDEN; j++) zbar[j] = 0.0f;
        // Region boundaries for this S: bounds[i] = (i*S)/REGION_GRID (integer division floors
        // for non-negative operands), so region row/col counts are (bounds[i+1]-bounds[i]).
        const int b1 = S / REGION_GRID;
        const int b2 = (2 * S) / REGION_GRID;
        const int bounds[REGION_GRID + 1] = {0, b1, b2, S};
        if (headVersion == 3) {
            for (int r = 0; r < REGION_HIDDEN; r++) zreg[r] = 0.0f;
        }
        for (int p = 0; p < P; p++) {
            const int zBase = (b * P + p) * LOCAL_HIDDEN;
            for (int j = 0; j < LOCAL_HIDDEN; j++) zbar[j] += zBuf[zBase + j];
            ownMean += ownership[b * P + p];
            if (headVersion == 3) {
                const int y = p / S;
                const int x = p % S;
                const int ri = (y < b1) ? 0 : ((y < b2) ? 1 : 2);
                const int ci = (x < b1) ? 0 : ((x < b2) ? 1 : 2);
                const int region = ri * REGION_GRID + ci;
                const int regionBase = region * LOCAL_HIDDEN;
                for (int j = 0; j < LOCAL_HIDDEN; j++) zreg[regionBase + j] += zBuf[zBase + j];
            }
        }
        for (int j = 0; j < LOCAL_HIDDEN; j++) zbar[j] /= float(P);
        ownMean /= float(P);
        if (headVersion == 3) {
            for (int ri = 0; ri < REGION_GRID; ri++) {
                const int rows = bounds[ri + 1] - bounds[ri];
                for (int ci = 0; ci < REGION_GRID; ci++) {
                    const int cols = bounds[ci + 1] - bounds[ci];
                    const float area = float(rows * cols);
                    const int regionBase = (ri * REGION_GRID + ci) * LOCAL_HIDDEN;
                    for (int j = 0; j < LOCAL_HIDDEN; j++) zreg[regionBase + j] /= area;
                }
            }
        }
    }

    float acc[GLOBAL_HIDDEN];
    for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] = bglobal[j];

    for (int c = 0; c < C; c++) {
        const float mval = mIn[b * C + c];
        const int row = c * GLOBAL_HIDDEN;
        for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += mval * Wglobal[row + j];
    }
    for (int c = 0; c < C; c++) {
        const float vval = vIn[b * C + c];
        const int row = (C + c) * GLOBAL_HIDDEN;
        for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += vval * Wglobal[row + j];
    }
    if (headVersion == 1) {
        for (int i = 0; i < 4; i++) {
            const float gval = globalIn[b * 4 + i];
            const int row = (2 * C + i) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += gval * Wglobal[row + j];
        }
    } else if (headVersion == 2) {
        for (int jz = 0; jz < LOCAL_HIDDEN; jz++) {
            const float zval = zbar[jz];
            const int row = (2 * C + jz) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += zval * Wglobal[row + j];
        }
        {
            const int row = (2 * C + LOCAL_HIDDEN) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += ownMean * Wglobal[row + j];
        }
        for (int i = 0; i < 4; i++) {
            const float gval = globalIn[b * 4 + i];
            const int row = (2 * C + LOCAL_HIDDEN + 1 + i) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += gval * Wglobal[row + j];
        }
    } else { // headVersion == 3: concat(m, v, zbar, zreg, ownMean, global)
        for (int jz = 0; jz < LOCAL_HIDDEN; jz++) {
            const float zval = zbar[jz];
            const int row = (2 * C + jz) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += zval * Wglobal[row + j];
        }
        for (int jr = 0; jr < REGION_HIDDEN; jr++) {
            const float zval = zreg[jr];
            const int row = (2 * C + LOCAL_HIDDEN + jr) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += zval * Wglobal[row + j];
        }
        {
            const int row = (2 * C + LOCAL_HIDDEN + REGION_HIDDEN) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += ownMean * Wglobal[row + j];
        }
        for (int i = 0; i < 4; i++) {
            const float gval = globalIn[b * 4 + i];
            const int row = (2 * C + LOCAL_HIDDEN + REGION_HIDDEN + 1 + i) * GLOBAL_HIDDEN;
            for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] += gval * Wglobal[row + j];
        }
    }
    for (int j = 0; j < GLOBAL_HIDDEN; j++) acc[j] = max(0.0f, acc[j]);

    float pass = bpass[0];
    float score = bscore[0];
    float wdl0 = bwdl[0], wdl1 = bwdl[1], wdl2 = bwdl[2];
    for (int j = 0; j < GLOBAL_HIDDEN; j++) {
        pass += acc[j] * Wpass[j];
        score += acc[j] * Wscore[j];
        wdl0 += acc[j] * Wwdl[j * 3 + 0];
        wdl1 += acc[j] * Wwdl[j * 3 + 1];
        wdl2 += acc[j] * Wwdl[j * 3 + 2];
    }
    policyLogits[b * (P + 1) + P] = pass;
    scoreMean[b] = score;
    wdlLogits[b * 3 + 0] = wdl0;
    wdlLogits[b * 3 + 1] = wdl1;
    wdlLogits[b * 3 + 2] = wdl2;
}
