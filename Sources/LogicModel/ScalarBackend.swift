import Foundation

/// CPU scalar reference backend (docs/spec/01-network.md §5). One `UInt8` per activation,
/// ping-pong buffers per layer, truth-table lookup per gate. This is the golden oracle that all
/// faster backends must match bit-for-bit; it is intentionally unoptimised.
public struct ScalarBackend: LogicBackend {
    public let model: LogicModelData
    public let name = "cpu-scalar"

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
        let layers = layerOutputs(features: features)
        return try Heads.evaluate(model: model, lastLayer: layers[model.layers - 1], features: features)
    }

    /// Output bits of every logic layer, each `[B,S,S,C]` in `(((b*S+y)*S+x)*C+c)` order.
    /// Exposed for per-layer parity tests.
    public func layerOutputs(features: FeatureBatch) -> [[UInt8]] {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let inputC = FeatureLayout.spatialChannels
        var outputs: [[UInt8]] = []
        var prev = features.spatial
        var prevC = inputC
        for l in 0 ..< model.layers {
            var out = [UInt8](repeating: 0, count: B * S * S * C)
            let layerWiring = model.wiring[l]
            let layerGates = model.gates[l]
            let layerTables = model.gateTables[l]
            let arity = model.manifest.gateArity
            features.spatial.withUnsafeBufferPointer { input in
                prev.withUnsafeBufferPointer { prevBuf in
                    out.withUnsafeMutableBufferPointer { outBuf in
                        for b in 0 ..< B {
                            for y in 0 ..< S {
                                for x in 0 ..< S {
                                    let outBase = ((b * S + y) * S + x) * C
                                    for c in 0 ..< C {
                                        let refs = layerWiring[c]
                                        if arity == 2 {
                                            let a = read(refs[0], b, x, y, S, input, inputC, prevBuf, prevC)
                                            let bb = read(refs[1], b, x, y, S, input, inputC, prevBuf, prevC)
                                            let row = 2 * a + bb
                                            outBuf[outBase + c] = (layerGates[c] >> row) & 1
                                        } else {
                                            var row = 0
                                            for r in refs {
                                                let bit = read(r, b, x, y, S, input, inputC, prevBuf, prevC)
                                                row = row * 2 + Int(bit)
                                            }
                                            outBuf[outBase + c] = UInt8((layerTables[c] >> row) & 1)
                                        }
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
    private func read(
        _ r: GateReference, _ b: Int, _ x: Int, _ y: Int, _ S: Int,
        _ input: UnsafeBufferPointer<UInt8>, _ inputC: Int,
        _ prev: UnsafeBufferPointer<UInt8>, _ prevC: Int
    ) -> UInt8 {
        let xx = x + Int(r.dx)
        let yy = y + Int(r.dy)
        guard xx >= 0, xx < S, yy >= 0, yy < S else { return 0 }
        if r.bank == 1 {
            return input[((b * S + yy) * S + xx) * inputC + Int(r.channel)]
        }
        return prev[((b * S + yy) * S + xx) * prevC + Int(r.channel)]
    }
}
