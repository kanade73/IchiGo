import Foundation

/// FP32 heads (docs/spec/01-network.md §4), explicit loops, Float accumulation.
/// `lastLayer` is `[B,S,S,C]` bits; outputs are to-move perspective.
public enum Heads {
    public static func evaluate(model: LogicModelData, lastLayer h: [UInt8], features: FeatureBatch) throws -> RawBatch {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let P = S * S
        let H1 = ModelManifest.localHidden
        let H2 = ModelManifest.globalHidden
        let inLocal = 3 * C + 4
        let headVersion = model.manifest.headVersion
        let inGlobal = ModelManifest.globalInputSize(channels: C, headVersion: headVersion)
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
        var zbar = [Float](repeating: 0, count: H1)   // mean_xy(z_xy), headVersion 2
        var ownMean: Float = 0                         // mean_xy(ownership_xy), headVersion 2

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
            for c in 0 ..< C { ug[c] = m[c]; ug[C + c] = v[c] }
            if headVersion == 1 {
                for i in 0 ..< 4 { ug[2 * C + i] = g[i] }
            } else {
                for j in 0 ..< H1 { ug[2 * C + j] = zbar[j] / Float(P) }
                ug[2 * C + H1] = ownMean / Float(P)
                for i in 0 ..< 4 { ug[2 * C + H1 + 1 + i] = g[i] }
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
