import Foundation

/// FP32 heads (docs/spec/01-network.md §4), explicit loops, Float accumulation.
/// `lastLayer` is `[B,S,S,C]` bits; outputs are to-move perspective.
public enum Heads {
    /// Integer partition boundaries of a `size`-point axis into 3 contiguous, nearly equal ranges
    /// (docs/spec/01-network.md §4, headVersion 3): part `i` covers `[bounds[i], bounds[i+1])`.
    /// `bounds[i] = (i*size)/3` (integer division == floor for non-negative operands), matching
    /// `ichigo_train.model.region_bounds`. Shared by `Heads.evaluate` and
    /// `Heads.evaluateAccelerated`.
    static func regionBounds(_ size: Int) -> [Int] {
        [0, size / ModelManifest.regionGrid, (2 * size) / ModelManifest.regionGrid, size]
    }

    public static func evaluate(model: LogicModelData, lastLayer h: [UInt8], features: FeatureBatch) throws -> RawBatch {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let P = S * S
        let H1 = ModelManifest.localHidden
        let H2 = ModelManifest.globalHidden
        let regionHidden = ModelManifest.regionHidden
        let inLocal = 3 * C + 4
        let headVersion = model.manifest.headVersion
        let inGlobal = ModelManifest.globalInputSize(channels: C, headVersion: headVersion)

        // headVersion 3: region index (0..2) per row/column and the (row-count*col-count) area of
        // each of the 9 regions, computed once for this board size.
        let bounds = regionBounds(S)
        var regionOfIndex = [Int](repeating: 0, count: S)
        for r in 0 ..< ModelManifest.regionGrid {
            for v in bounds[r] ..< bounds[r + 1] { regionOfIndex[v] = r }
        }
        var regionArea = [Float](repeating: 0, count: ModelManifest.regionCount)
        for ri in 0 ..< ModelManifest.regionGrid {
            for ci in 0 ..< ModelManifest.regionGrid {
                regionArea[ri * ModelManifest.regionGrid + ci] = Float((bounds[ri + 1] - bounds[ri]) * (bounds[ci + 1] - bounds[ci]))
            }
        }
        let Wlocal = model.head("Wlocal"), blocal = model.head("blocal")
        let Wpolicy = model.head("Wpolicy"), bpolicy = model.head("bpolicy")
        let Wowner = model.head("Wowner"), bowner = model.head("bowner")
        let Wglobal = model.head("Wglobal"), bglobal = model.head("bglobal")
        let Wpass = model.head("Wpass"), bpass = model.head("bpass")
        let Wwdl = model.head("Wwdl"), bwdl = model.head("bwdl")
        let Wscore = model.head("Wscore"), bscore = model.head("bscore")

        var policyLogits = [Float](repeating: 0, count: B * (P + 1))
        var wdlLogits = [Float](repeating: 0, count: B * 3)
        var scoreMean = [Float](repeating: 0, count: B)
        var ownership = [Float](repeating: 0, count: B * P)

        var u = [Float](repeating: 0, count: inLocal)
        var z = [Float](repeating: 0, count: H1)
        var ug = [Float](repeating: 0, count: inGlobal)
        var zg = [Float](repeating: 0, count: H2)
        var m = [Float](repeating: 0, count: C)
        var v = [Float](repeating: 0, count: C)
        var zbar = [Float](repeating: 0, count: H1)          // mean_xy(z_xy), headVersion 2, 3
        var ownMean: Float = 0                                // mean_xy(ownership_xy), headVersion 2, 3
        var zreg = [Float](repeating: 0, count: regionHidden) // per-3x3-region mean_xy(z_xy), headVersion 3

        for b in 0 ..< B {
            // m_c = mean_xy(h_c), v_c = max_xy(h_c)
            for c in 0 ..< C { m[c] = 0; v[c] = 0 }
            for p in 0 ..< P {
                let base = (b * P + p) * C
                for c in 0 ..< C {
                    let val = Float(h[base + c])
                    m[c] += val
                    if val > v[c] { v[c] = val }
                }
            }
            for c in 0 ..< C { m[c] /= Float(P) }
            let g = Array(features.global[(b * 4) ..< (b * 4 + 4)])
            // shared tail of u: (m, v, global)
            for c in 0 ..< C { u[C + c] = m[c]; u[2 * C + c] = v[c] }
            for i in 0 ..< 4 { u[3 * C + i] = g[i] }
            for j in 0 ..< H1 { zbar[j] = 0 }
            for r in 0 ..< regionHidden { zreg[r] = 0 }
            ownMean = 0
            for p in 0 ..< P {
                let base = (b * P + p) * C
                for c in 0 ..< C { u[c] = Float(h[base + c]) }
                for j in 0 ..< H1 {
                    var acc = blocal[j]
                    for i in 0 ..< inLocal { acc += u[i] * Wlocal[i * H1 + j] }
                    z[j] = max(0, acc)
                    zbar[j] += z[j]
                }
                let y = p / S, x = p % S
                let region = regionOfIndex[y] * ModelManifest.regionGrid + regionOfIndex[x]
                let regionBase = region * H1
                for j in 0 ..< H1 { zreg[regionBase + j] += z[j] }
                var pol = bpolicy[0]
                var own = bowner[0]
                for j in 0 ..< H1 {
                    pol += z[j] * Wpolicy[j]
                    own += z[j] * Wowner[j]
                }
                policyLogits[b * (P + 1) + p] = pol
                let o = tanh(own)
                ownership[b * P + p] = o
                ownMean += o
            }
            for region in 0 ..< ModelManifest.regionCount {
                let area = regionArea[region]
                let regionBase = region * H1
                for j in 0 ..< H1 { zreg[regionBase + j] /= area }
            }
            for c in 0 ..< C { ug[c] = m[c]; ug[C + c] = v[c] }
            switch headVersion {
            case 1:
                for i in 0 ..< 4 { ug[2 * C + i] = g[i] }
            case 2:
                for j in 0 ..< H1 { ug[2 * C + j] = zbar[j] / Float(P) }
                ug[2 * C + H1] = ownMean / Float(P)
                for i in 0 ..< 4 { ug[2 * C + H1 + 1 + i] = g[i] }
            default: // headVersion 3
                for j in 0 ..< H1 { ug[2 * C + j] = zbar[j] / Float(P) }
                for r in 0 ..< regionHidden { ug[2 * C + H1 + r] = zreg[r] } // already normalised by area
                ug[2 * C + H1 + regionHidden] = ownMean / Float(P)
                for i in 0 ..< 4 { ug[2 * C + H1 + regionHidden + 1 + i] = g[i] }
            }
            for j in 0 ..< H2 {
                var acc = bglobal[j]
                for i in 0 ..< inGlobal { acc += ug[i] * Wglobal[i * H2 + j] }
                zg[j] = max(0, acc)
            }
            var pass = bpass[0]
            var score = bscore[0]
            var wdl: [Float] = [bwdl[0], bwdl[1], bwdl[2]]
            for j in 0 ..< H2 {
                pass += zg[j] * Wpass[j]
                score += zg[j] * Wscore[j]
                wdl[0] += zg[j] * Wwdl[j * 3]
                wdl[1] += zg[j] * Wwdl[j * 3 + 1]
                wdl[2] += zg[j] * Wwdl[j * 3 + 2]
            }
            policyLogits[b * (P + 1) + P] = pass
            scoreMean[b] = score
            wdlLogits[b * 3] = wdl[0]; wdlLogits[b * 3 + 1] = wdl[1]; wdlLogits[b * 3 + 2] = wdl[2]
        }
        for (name, arr) in [("policyLogits", policyLogits), ("wdlLogits", wdlLogits), ("scoreMean", scoreMean), ("ownership", ownership)] {
            guard arr.allSatisfy({ $0.isFinite }) else { throw LogicModelError.nonFiniteOutput(name) }
        }
        return RawBatch(boardSize: S, batch: B, policyLogits: policyLogits, wdlLogits: wdlLogits, scoreMean: scoreMean, ownership: ownership)
    }
}
