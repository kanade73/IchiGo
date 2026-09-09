import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// A faster path through the same FP32 head math as `Heads.evaluate` (docs/spec/01-network.md
/// §4), used by every backend except `ScalarBackend` (which stays on the explicit-loop version
/// forever, per docs/spec/01-network.md §5: "最適化前の golden oracle として永久に保持").
/// `evaluateAccelerated` is NOT itself an oracle: it reassociates floating-point sums (batched
/// matmuls instead of one-point-at-a-time accumulation), so results only need to match
/// `Heads.evaluate` within docs/spec/05-validation.md §3 tolerance, not bit-for-bit --
/// `HeadsAcceleratedTests` checks exactly that. On macOS this dispatches through `cblas_sgemm`
/// (Accelerate); everywhere else (guarded by `#if canImport(Accelerate)`) it falls back to
/// `Heads.evaluate` unchanged, so the fallback is always correct even if not faster.
public extension Heads {
    static func evaluateAccelerated(model: LogicModelData, lastLayer h: [UInt8], features: FeatureBatch) throws -> RawBatch {
        #if canImport(Accelerate)
        return try evaluateBLAS(model: model, lastLayer: h, features: features)
        #else
        return try evaluate(model: model, lastLayer: h, features: features)
        #endif
    }
}

#if canImport(Accelerate)
extension Heads {
    /// Row-major `C[M,N] = alpha*A[M,K]*B[K,N] + beta*C[M,N]`.
    static func sgemm(
        _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ c: UnsafeMutablePointer<Float>,
        m: Int, k: Int, n: Int, alpha: Float = 1, beta: Float = 0
    ) {
        guard m > 0, k > 0, n > 0 else { return }
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(m), Int32(n), Int32(k), alpha,
            a, Int32(k), b, Int32(n), beta, c, Int32(n)
        )
    }

    static func evaluateBLAS(model: LogicModelData, lastLayer h: [UInt8], features: FeatureBatch) throws -> RawBatch {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let P = S * S
        let H1 = ModelManifest.localHidden   // 64, fixed regardless of channels/headVersion
        let H2 = ModelManifest.globalHidden  // 128
        let regionHidden = ModelManifest.regionHidden // 576 = 9*64, headVersion 3 only
        let headVersion = model.manifest.headVersion

        let Wlocal = model.head("Wlocal"), blocal = model.head("blocal")
        let Wpolicy = model.head("Wpolicy"), bpolicy = model.head("bpolicy")[0]
        let Wowner = model.head("Wowner"), bowner = model.head("bowner")[0]
        let Wglobal = model.head("Wglobal"), bglobal = model.head("bglobal")
        let Wpass = model.head("Wpass"), bpass = model.head("bpass")[0]
        let Wwdl = model.head("Wwdl"), bwdl = model.head("bwdl")
        let Wscore = model.head("Wscore"), bscore = model.head("bscore")[0]

        // h as Float, [B*P,C] row-major (a straight cast, contiguous with the input layout).
        var Hf = [Float](repeating: 0, count: B * P * C)
        for i in 0 ..< (B * P * C) { Hf[i] = Float(h[i]) }

        // m[B,C] = mean_xy(h), v[B,C] = max_xy(h).
        var M = [Float](repeating: 0, count: B * C)
        var V = [Float](repeating: 0, count: B * C)
        Hf.withUnsafeBufferPointer { hf in
            M.withUnsafeMutableBufferPointer { mBuf in
                V.withUnsafeMutableBufferPointer { vBuf in
                    for b in 0 ..< B {
                        let base = b * P * C
                        for p in 0 ..< P {
                            let row = base + p * C
                            for c in 0 ..< C {
                                let val = hf[row + c]
                                mBuf[b * C + c] += val
                                if val > vBuf[b * C + c] { vBuf[b * C + c] = val }
                            }
                        }
                        for c in 0 ..< C { mBuf[b * C + c] /= Float(P) }
                    }
                }
            }
        }

        // Local projection, split by input segment so none of h/m/v/global needs broadcasting
        // into a materialised [B*P, 3C+4] matrix: z_xy = ReLU(H@Wlocal_h + M@Wlocal_m (per-b,
        // broadcast over p) + V@Wlocal_v (broadcast) + G@Wlocal_g (broadcast) + blocal).
        // Wlocal is [3C+4,64] row-major, so each segment is a contiguous row range.
        var HZ = [Float](repeating: 0, count: B * P * H1)
        var MZ = [Float](repeating: 0, count: B * H1)
        var VZ = [Float](repeating: 0, count: B * H1)
        var GZ = [Float](repeating: 0, count: B * H1)
        Wlocal.withUnsafeBufferPointer { w in
            Hf.withUnsafeBufferPointer { hf in
                HZ.withUnsafeMutableBufferPointer { hz in sgemm(hf.baseAddress!, w.baseAddress!, hz.baseAddress!, m: B * P, k: C, n: H1) }
            }
            M.withUnsafeBufferPointer { m in
                MZ.withUnsafeMutableBufferPointer { mz in sgemm(m.baseAddress!, w.baseAddress! + C * H1, mz.baseAddress!, m: B, k: C, n: H1) }
            }
            V.withUnsafeBufferPointer { v in
                VZ.withUnsafeMutableBufferPointer { vz in sgemm(v.baseAddress!, w.baseAddress! + 2 * C * H1, vz.baseAddress!, m: B, k: C, n: H1) }
            }
            features.global.withUnsafeBufferPointer { g in
                GZ.withUnsafeMutableBufferPointer { gz in sgemm(g.baseAddress!, w.baseAddress! + 3 * C * H1, gz.baseAddress!, m: B, k: 4, n: H1) }
            }
        }
        var z = [Float](repeating: 0, count: B * P * H1)
        z.withUnsafeMutableBufferPointer { zBuf in
            HZ.withUnsafeBufferPointer { hz in
                MZ.withUnsafeBufferPointer { mz in
                    VZ.withUnsafeBufferPointer { vz in
                        GZ.withUnsafeBufferPointer { gz in
                            blocal.withUnsafeBufferPointer { bl in
                                for b in 0 ..< B {
                                    for p in 0 ..< P {
                                        let row = (b * P + p) * H1
                                        for j in 0 ..< H1 {
                                            let acc = hz[row + j] + mz[b * H1 + j] + vz[b * H1 + j] + gz[b * H1 + j] + bl[j]
                                            zBuf[row + j] = max(0, acc)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // policy_xy = z_xy@Wpolicy + bpolicy, ownership_xy = tanh(z_xy@Wowner + bowner).
        var policyPerPoint = [Float](repeating: 0, count: B * P)
        var ownership = [Float](repeating: 0, count: B * P)
        z.withUnsafeBufferPointer { zb in
            Wpolicy.withUnsafeBufferPointer { wp in
                policyPerPoint.withUnsafeMutableBufferPointer { pp in sgemm(zb.baseAddress!, wp.baseAddress!, pp.baseAddress!, m: B * P, k: H1, n: 1) }
            }
            Wowner.withUnsafeBufferPointer { wo in
                ownership.withUnsafeMutableBufferPointer { own in sgemm(zb.baseAddress!, wo.baseAddress!, own.baseAddress!, m: B * P, k: H1, n: 1) }
            }
        }
        var policyLogits = [Float](repeating: 0, count: B * (P + 1))
        for b in 0 ..< B {
            for p in 0 ..< P { policyLogits[b * (P + 1) + p] = policyPerPoint[b * P + p] + bpolicy }
        }
        for i in 0 ..< (B * P) { ownership[i] = tanh(ownership[i] + bowner) }

        // zbar = mean_xy(z_xy), ownMean = mean_xy(ownership) -- headVersion 2, 3 only.
        var zbar = [Float](repeating: 0, count: B * H1)
        var ownMean = [Float](repeating: 0, count: B)
        if headVersion == 2 || headVersion == 3 {
            z.withUnsafeBufferPointer { zb in
                zbar.withUnsafeMutableBufferPointer { zbarBuf in
                    for b in 0 ..< B {
                        for p in 0 ..< P {
                            let row = (b * P + p) * H1
                            for j in 0 ..< H1 { zbarBuf[b * H1 + j] += zb[row + j] }
                        }
                        for j in 0 ..< H1 { zbarBuf[b * H1 + j] /= Float(P) }
                    }
                }
            }
            for b in 0 ..< B {
                var s: Float = 0
                for p in 0 ..< P { s += ownership[b * P + p] }
                ownMean[b] = s / Float(P)
            }
        }

        // zreg = per-3x3-region mean_xy(z_xy) -- headVersion 3 only (docs/spec/01-network.md §4).
        // Plain nested loops (not BLAS), same style as the m/v reduction above: this is a spatial
        // reduce, not a projection.
        var zreg = [Float](repeating: 0, count: B * regionHidden)
        if headVersion == 3 {
            let bounds = Heads.regionBounds(S)
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
            z.withUnsafeBufferPointer { zb in
                zreg.withUnsafeMutableBufferPointer { zregBuf in
                    for b in 0 ..< B {
                        for y in 0 ..< S {
                            let ri = regionOfIndex[y]
                            for x in 0 ..< S {
                                let ci = regionOfIndex[x]
                                let region = ri * ModelManifest.regionGrid + ci
                                let p = y * S + x
                                let zRow = (b * P + p) * H1
                                let outBase = b * regionHidden + region * H1
                                for j in 0 ..< H1 { zregBuf[outBase + j] += zb[zRow + j] }
                            }
                        }
                        for region in 0 ..< ModelManifest.regionCount {
                            let area = regionArea[region]
                            let outBase = b * regionHidden + region * H1
                            for j in 0 ..< H1 { zregBuf[outBase + j] /= area }
                        }
                    }
                }
            }
        }

        // Global projection, same broadcast-segment trick as the local one.
        var ZG = [Float](repeating: 0, count: B * H2)
        ZG.withUnsafeMutableBufferPointer { zg in
            for b in 0 ..< B { for j in 0 ..< H2 { zg[b * H2 + j] = bglobal[j] } }
        }
        Wglobal.withUnsafeBufferPointer { w in
            M.withUnsafeBufferPointer { m in
                ZG.withUnsafeMutableBufferPointer { zg in sgemm(m.baseAddress!, w.baseAddress!, zg.baseAddress!, m: B, k: C, n: H2, beta: 1) }
            }
            V.withUnsafeBufferPointer { v in
                ZG.withUnsafeMutableBufferPointer { zg in sgemm(v.baseAddress!, w.baseAddress! + C * H2, zg.baseAddress!, m: B, k: C, n: H2, beta: 1) }
            }
            if headVersion == 1 {
                features.global.withUnsafeBufferPointer { g in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(g.baseAddress!, w.baseAddress! + 2 * C * H2, zg.baseAddress!, m: B, k: 4, n: H2, beta: 1) }
                }
            } else if headVersion == 2 {
                zbar.withUnsafeBufferPointer { zbarBuf in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(zbarBuf.baseAddress!, w.baseAddress! + 2 * C * H2, zg.baseAddress!, m: B, k: H1, n: H2, beta: 1) }
                }
                ownMean.withUnsafeBufferPointer { ownBuf in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(ownBuf.baseAddress!, w.baseAddress! + (2 * C + H1) * H2, zg.baseAddress!, m: B, k: 1, n: H2, beta: 1) }
                }
                features.global.withUnsafeBufferPointer { g in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(g.baseAddress!, w.baseAddress! + (2 * C + H1 + 1) * H2, zg.baseAddress!, m: B, k: 4, n: H2, beta: 1) }
                }
            } else { // headVersion 3: concat(m, v, zbar, zreg, ownMean, global)
                zbar.withUnsafeBufferPointer { zbarBuf in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(zbarBuf.baseAddress!, w.baseAddress! + 2 * C * H2, zg.baseAddress!, m: B, k: H1, n: H2, beta: 1) }
                }
                zreg.withUnsafeBufferPointer { zregBuf in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(zregBuf.baseAddress!, w.baseAddress! + (2 * C + H1) * H2, zg.baseAddress!, m: B, k: regionHidden, n: H2, beta: 1) }
                }
                ownMean.withUnsafeBufferPointer { ownBuf in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(ownBuf.baseAddress!, w.baseAddress! + (2 * C + H1 + regionHidden) * H2, zg.baseAddress!, m: B, k: 1, n: H2, beta: 1) }
                }
                features.global.withUnsafeBufferPointer { g in
                    ZG.withUnsafeMutableBufferPointer { zg in sgemm(g.baseAddress!, w.baseAddress! + (2 * C + H1 + regionHidden + 1) * H2, zg.baseAddress!, m: B, k: 4, n: H2, beta: 1) }
                }
            }
        }
        for i in 0 ..< (B * H2) { ZG[i] = max(0, ZG[i]) }

        var passLogit = [Float](repeating: 0, count: B)
        var scoreMean = [Float](repeating: 0, count: B)
        var wdlLogits = [Float](repeating: 0, count: B * 3)
        ZG.withUnsafeBufferPointer { zg in
            Wpass.withUnsafeBufferPointer { wp in
                passLogit.withUnsafeMutableBufferPointer { pl in sgemm(zg.baseAddress!, wp.baseAddress!, pl.baseAddress!, m: B, k: H2, n: 1) }
            }
            Wscore.withUnsafeBufferPointer { ws in
                scoreMean.withUnsafeMutableBufferPointer { sm in sgemm(zg.baseAddress!, ws.baseAddress!, sm.baseAddress!, m: B, k: H2, n: 1) }
            }
            Wwdl.withUnsafeBufferPointer { ww in
                wdlLogits.withUnsafeMutableBufferPointer { wdl in sgemm(zg.baseAddress!, ww.baseAddress!, wdl.baseAddress!, m: B, k: H2, n: 3) }
            }
        }
        for b in 0 ..< B {
            policyLogits[b * (P + 1) + P] = passLogit[b] + bpass
            scoreMean[b] += bscore
            wdlLogits[b * 3] += bwdl[0]; wdlLogits[b * 3 + 1] += bwdl[1]; wdlLogits[b * 3 + 2] += bwdl[2]
        }

        for (name, arr) in [("policyLogits", policyLogits), ("wdlLogits", wdlLogits), ("scoreMean", scoreMean), ("ownership", ownership)] {
            guard arr.allSatisfy({ $0.isFinite }) else { throw LogicModelError.nonFiniteOutput(name) }
        }
        return RawBatch(boardSize: S, batch: B, policyLogits: policyLogits, wdlLogits: wdlLogits, scoreMean: scoreMean, ownership: ownership)
    }
}
#endif
