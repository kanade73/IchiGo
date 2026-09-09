import Foundation

/// CPU packed backend (docs/spec/01-network.md §5, docs/spec/04-tasks.md T23). Batches are packed
/// 32-at-a-time into `UInt32` lanes (`PackBits`); each output bit is produced by the four-term
/// formula from §5 applied to whole words, so one word covers up to 32 samples per gate
/// evaluation instead of one. Must match `ScalarBackend` bit-for-bit once unpacked -- this is a
/// storage/throughput change only, never a numeric approximation.
public struct PackedCPUBackend: LogicBackend {
    public let model: LogicModelData
    public let name = "cpu-packed"

    public init(model: LogicModelData) {
        self.model = model
    }

    public func evaluate(features: FeatureBatch) async throws -> RawBatch {
        try evaluateSync(features: features)
    }

    public func evaluateSync(features: FeatureBatch) throws -> RawBatch {
        guard model.manifest.boardSizes.contains(features.boardSize) else {
            throw LogicModelError.invalidInput("model does not support board size \(features.boardSize)")
        }
        if features.batch == 0 {
            return RawBatch(boardSize: features.boardSize, batch: 0, policyLogits: [], wdlLogits: [], scoreMean: [], ownership: [])
        }
        let S = features.boardSize, B = features.batch, C = model.channels
        let packed = packedLayerOutputs(features: features)
        let lastBits = PackBits.unpack(packed[model.layers - 1], boardSize: S, batch: B, channels: C)
        return try Heads.evaluateAccelerated(model: model, lastLayer: lastBits, features: features)
    }

    /// Output bits of every logic layer, unpacked to `[B,S,S,C]` `(((b*S+y)*S+x)*C+c)` order --
    /// bit-for-bit identical to `ScalarBackend.layerOutputs`. Exposed for per-layer parity tests.
    public func layerOutputs(features: FeatureBatch) -> [[UInt8]] {
        let S = features.boardSize, B = features.batch, C = model.channels
        return packedLayerOutputs(features: features).map { PackBits.unpack($0, boardSize: S, batch: B, channels: C) }
    }

    /// Output bits of every logic layer, packed `[G,S,S,C]` `UInt32` (bit k = sample
    /// `group*32+k`), `G = ceil(B/32)`. Exposed so `MetalPackedBackend`'s parity tests and the
    /// benchmark can time the packed gate stage without paying for CPU unpack.
    public func packedLayerOutputs(features: FeatureBatch) -> [[UInt32]] {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let inputC = FeatureLayout.spatialChannels
        let G = PackBits.groupCount(batch: B)
        let inputPacked = PackBits.pack(features.spatial, boardSize: S, batch: B, channels: inputC)
        var outputs: [[UInt32]] = []
        outputs.reserveCapacity(model.layers)
        var prev = inputPacked
        var prevC = inputC
        for l in 0 ..< model.layers {
            var out = [UInt32](repeating: 0, count: G * S * S * C)
            let layerWiring = model.wiring[l]
            let layerGates = model.gates[l]
            inputPacked.withUnsafeBufferPointer { input in
                prev.withUnsafeBufferPointer { prevBuf in
                    out.withUnsafeMutableBufferPointer { outBuf in
                        for group in 0 ..< G {
                            let valid = PackBits.validMask(batch: B, group: group)
                            for y in 0 ..< S {
                                for x in 0 ..< S {
                                    let outBase = ((group * S + y) * S + x) * C
                                    for c in 0 ..< C {
                                        let refs = layerWiring[c]
                                        let a = PackedCPUBackend.read(refs[0], group, x, y, S, input, inputC, prevBuf, prevC)
                                        let b = PackedCPUBackend.read(refs[1], group, x, y, S, input, inputC, prevBuf, prevC)
                                        let g = layerGates[c]
                                        var bits: UInt32 = 0
                                        if g & 1 != 0 { bits |= ~a & ~b }
                                        if g & 2 != 0 { bits |= ~a & b }
                                        if g & 4 != 0 { bits |= a & ~b }
                                        if g & 8 != 0 { bits |= a & b }
                                        outBuf[outBase + c] = bits & valid
                                    }
                                }
                            }
                        }
                    }
                }
            }
            outputs.append(out)
            prev = out
            prevC = C
        }
        return outputs
    }

    @inline(__always)
    private static func read(
        _ r: GateReference, _ group: Int, _ x: Int, _ y: Int, _ S: Int,
        _ input: UnsafeBufferPointer<UInt32>, _ inputC: Int,
        _ prev: UnsafeBufferPointer<UInt32>, _ prevC: Int
    ) -> UInt32 {
        let xx = x + Int(r.dx)
        let yy = y + Int(r.dy)
        guard xx >= 0, xx < S, yy >= 0, yy < S else { return 0 }
        if r.bank == 1 {
            return input[((group * S + yy) * S + xx) * inputC + Int(r.channel)]
        }
        return prev[((group * S + yy) * S + xx) * prevC + Int(r.channel)]
    }
}
