import Foundation
import LogicModel
import Metal

private extension Duration {
    var milliseconds: Double { Double(components.seconds) * 1000 + Double(components.attoseconds) * 1e-15 }
}

/// Metal-packed backend: packed (batch-direction `UInt32`) gate layers *and* Metal heads
/// (docs/spec/01-network.md §4-5, docs/spec/04-tasks.md T23/T24). Gate layers use the same
/// `[G,S,S,C]` packed layout as `PackedCPUBackend` (`PackBits`), one dispatch per layer; the head
/// kernels (`heads_reduce`/`heads_local`/`heads_global`) read the last packed gate layer directly
/// (no CPU unpack in between) and are encoded into the *same* `MTLCommandBuffer` as the gate
/// dispatches, so `evaluate` commits and waits exactly once. Head kernels are compiled with
/// fast-math off (`MTLCompileOptions.fastMathEnabled = false`) and run entirely in FP32, matching
/// `Heads.evaluate`'s numeric contract (docs/spec/05-validation.md §3 tolerance).
///
/// Only the final outputs (`policyLogits`, `wdlLogits`, `scoreMean`, `ownership`) are read back;
/// the intermediate `m`/`v`/`z` buffers never leave the device.
public final class MetalPackedBackend: LogicBackend, @unchecked Sendable {
    public let model: LogicModelData
    public let name = "metal-packed"

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let gatePipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let localPipeline: MTLComputePipelineState
    private let globalPipeline: MTLComputePipelineState

    // Head weight buffers, uploaded once at init and reused across `evaluate` calls.
    private let headBuffers: [String: MTLBuffer]

    public var currentAllocatedBytes: UInt64 { UInt64(device.currentAllocatedSize) }
    public var deviceName: String { device.name }

    public init(model: LogicModelData) throws {
        guard model.manifest.gateArity == 2 else {
            throw LogicModelError.backendUnavailable("Metal packed backend does not support gate arity \(model.manifest.gateArity); use cpu-packed or cpu")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LogicModelError.backendUnavailable("no Metal device on this host")
        }
        guard let queue = device.makeCommandQueue() else {
            throw LogicModelError.backendUnavailable("cannot create a Metal command queue")
        }
        let source = try MetalPackedBackend.kernelSource()
        let options = MTLCompileOptions()
        options.fastMathEnabled = false // docs/spec/04-tasks.md T24: "fast-mathを初期offにし"
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw LogicModelError.backendUnavailable("cannot compile logic_packed.metal/heads.metal: \(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw LogicModelError.backendUnavailable("\(name) function missing from compiled Metal library")
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw LogicModelError.backendUnavailable("cannot create Metal compute pipeline for \(name): \(error)")
            }
        }
        gatePipeline = try pipeline("logic_layer_packed")
        reducePipeline = try pipeline("heads_reduce")
        localPipeline = try pipeline("heads_local")
        globalPipeline = try pipeline("heads_global")

        var buffers: [String: MTLBuffer] = [:]
        for name in ModelManifest.headTensorNames {
            let values = model.head(name)
            guard let buf = values.withUnsafeBytes({ raw -> MTLBuffer? in
                guard raw.count > 0 else { return device.makeBuffer(length: 4, options: .storageModeShared) }
                return device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }) else {
                throw LogicModelError.backendUnavailable("cannot allocate Metal head buffer for \(name)")
            }
            buffers[name] = buf
        }
        headBuffers = buffers
        self.model = model
        self.device = device
        self.queue = queue
    }

    public func evaluate(features: FeatureBatch) async throws -> RawBatch {
        guard model.manifest.boardSizes.contains(features.boardSize) else {
            throw LogicModelError.invalidInput("model does not support board size \(features.boardSize)")
        }
        if features.batch == 0 {
            return RawBatch(boardSize: features.boardSize, batch: 0, policyLogits: [], wdlLogits: [], scoreMean: [], ownership: [])
        }

        let (commandBuffer, gateOutBuffers, _, gateKeepAlive) = try encodeGateLayers(features: features)
        let heads = try encodeHeads(commandBuffer: commandBuffer, lastLayerBuffer: gateOutBuffers[model.layers - 1], features: features)

        try await MetalBackend.commitAndWait(commandBuffer)
        _ = gateKeepAlive; _ = heads.keepAlive // kept alive across the await above

        return try heads.readBack()
    }

    /// Benchmark-only entry point (docs/spec/05-validation.md §7): times the gate and head stages
    /// separately by encoding them as two command buffers instead of one -- `evaluate` itself
    /// keeps gates+heads fused into a single command buffer/commit as T24 requires; this method
    /// exists purely so `ichigo benchmark` can report a `gate_ms`/`head_ms` split for this backend
    /// the same way it does for the others. Returns the identical `RawBatch` `evaluate` would.
    public func evaluateTimed(features: FeatureBatch) async throws -> (raw: RawBatch, gateMs: Double, headMs: Double) {
        guard model.manifest.boardSizes.contains(features.boardSize) else {
            throw LogicModelError.invalidInput("model does not support board size \(features.boardSize)")
        }
        if features.batch == 0 {
            return (RawBatch(boardSize: features.boardSize, batch: 0, policyLogits: [], wdlLogits: [], scoreMean: [], ownership: []), 0, 0)
        }
        let clock = ContinuousClock()
        var t0 = clock.now

        let (gateCommandBuffer, gateOutBuffers, _, gateKeepAlive) = try encodeGateLayers(features: features)
        try await MetalBackend.commitAndWait(gateCommandBuffer)
        _ = gateKeepAlive
        let gateMs = (clock.now - t0).milliseconds
        t0 = clock.now

        guard let headCommandBuffer = queue.makeCommandBuffer() else {
            throw LogicModelError.backendUnavailable("cannot create Metal command buffer")
        }
        let heads = try encodeHeads(commandBuffer: headCommandBuffer, lastLayerBuffer: gateOutBuffers[model.layers - 1], features: features)
        try await MetalBackend.commitAndWait(headCommandBuffer)
        _ = heads.keepAlive
        let headMs = (clock.now - t0).milliseconds

        return (try heads.readBack(), gateMs, headMs)
    }

    /// Encodes the three head kernels (reduce/local/global) onto `commandBuffer`, reading the
    /// packed last gate layer directly. Returns a closure that reads the outputs back once the
    /// command buffer this was encoded onto has completed, plus the buffers that must stay alive
    /// until then.
    private func encodeHeads(
        commandBuffer: MTLCommandBuffer, lastLayerBuffer: MTLBuffer, features: FeatureBatch
    ) throws -> (readBack: () throws -> RawBatch, keepAlive: [MTLBuffer]) {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let P = S * S
        let H1 = ModelManifest.localHidden
        guard let mBuf = device.makeBuffer(length: B * C * 4, options: .storageModeShared),
              let vBuf = device.makeBuffer(length: B * C * 4, options: .storageModeShared),
              let zBuf = device.makeBuffer(length: B * P * H1 * 4, options: .storageModeShared),
              let ownershipBuf = device.makeBuffer(length: B * P * 4, options: .storageModeShared),
              let policyBuf = device.makeBuffer(length: B * (P + 1) * 4, options: .storageModeShared),
              let wdlBuf = device.makeBuffer(length: B * 3 * 4, options: .storageModeShared),
              let scoreBuf = device.makeBuffer(length: B * 4, options: .storageModeShared),
              let globalBuf = features.global.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) })
        else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal head buffers")
        }

        var keepAlive: [MTLBuffer] = [mBuf, vBuf, zBuf, ownershipBuf, policyBuf, wdlBuf, scoreBuf, globalBuf]

        // heads_reduce: grid (C, B) -> m[B,C], v[B,C].
        let reduceDims: [Int32] = [Int32(S), Int32(C), Int32(B)]
        guard let reduceDimsBuf = reduceDims.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) }) else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal dims buffer")
        }
        keepAlive.append(reduceDimsBuf)
        guard let reduceEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LogicModelError.backendUnavailable("cannot create Metal compute encoder for heads_reduce")
        }
        reduceEncoder.setComputePipelineState(reducePipeline)
        reduceEncoder.setBuffer(lastLayerBuffer, offset: 0, index: 0)
        reduceEncoder.setBuffer(mBuf, offset: 0, index: 1)
        reduceEncoder.setBuffer(vBuf, offset: 0, index: 2)
        reduceEncoder.setBuffer(reduceDimsBuf, offset: 0, index: 3)
        reduceEncoder.dispatchThreads(MTLSize(width: C, height: B, depth: 1), threadsPerThreadgroup: MetalPackedBackend.threadgroupSize2D(reducePipeline, w: C, h: B))
        reduceEncoder.endEncoding()

        // heads_local: grid (S, S, B) -> per-point policy/ownership + zBuf.
        let localDims: [Int32] = [Int32(S), Int32(C), Int32(B)]
        guard let localDimsBuf = localDims.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) }) else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal dims buffer")
        }
        keepAlive.append(localDimsBuf)
        guard let localEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LogicModelError.backendUnavailable("cannot create Metal compute encoder for heads_local")
        }
        localEncoder.setComputePipelineState(localPipeline)
        localEncoder.setBuffer(lastLayerBuffer, offset: 0, index: 0)
        localEncoder.setBuffer(mBuf, offset: 0, index: 1)
        localEncoder.setBuffer(vBuf, offset: 0, index: 2)
        localEncoder.setBuffer(globalBuf, offset: 0, index: 3)
        localEncoder.setBuffer(headBuffers["Wlocal"]!, offset: 0, index: 4)
        localEncoder.setBuffer(headBuffers["blocal"]!, offset: 0, index: 5)
        localEncoder.setBuffer(headBuffers["Wpolicy"]!, offset: 0, index: 6)
        localEncoder.setBuffer(headBuffers["bpolicy"]!, offset: 0, index: 7)
        localEncoder.setBuffer(headBuffers["Wowner"]!, offset: 0, index: 8)
        localEncoder.setBuffer(headBuffers["bowner"]!, offset: 0, index: 9)
        localEncoder.setBuffer(policyBuf, offset: 0, index: 10)
        localEncoder.setBuffer(ownershipBuf, offset: 0, index: 11)
        localEncoder.setBuffer(zBuf, offset: 0, index: 12)
        localEncoder.setBuffer(localDimsBuf, offset: 0, index: 13)
        localEncoder.dispatchThreads(MTLSize(width: S, height: S, depth: B), threadsPerThreadgroup: MetalPackedBackend.threadgroupSize3D(localPipeline, s: S, z: B))
        localEncoder.endEncoding()

        // heads_global: grid (B) -> pass/wdl/score.
        let globalDims: [Int32] = [Int32(S), Int32(C), Int32(B), Int32(model.manifest.headVersion)]
        guard let globalDimsBuf = globalDims.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) }) else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal dims buffer")
        }
        keepAlive.append(globalDimsBuf)
        guard let globalEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw LogicModelError.backendUnavailable("cannot create Metal compute encoder for heads_global")
        }
        globalEncoder.setComputePipelineState(globalPipeline)
        globalEncoder.setBuffer(mBuf, offset: 0, index: 0)
        globalEncoder.setBuffer(vBuf, offset: 0, index: 1)
        globalEncoder.setBuffer(zBuf, offset: 0, index: 2)
        globalEncoder.setBuffer(ownershipBuf, offset: 0, index: 3)
        globalEncoder.setBuffer(globalBuf, offset: 0, index: 4)
        globalEncoder.setBuffer(headBuffers["Wglobal"]!, offset: 0, index: 5)
        globalEncoder.setBuffer(headBuffers["bglobal"]!, offset: 0, index: 6)
        globalEncoder.setBuffer(headBuffers["Wpass"]!, offset: 0, index: 7)
        globalEncoder.setBuffer(headBuffers["bpass"]!, offset: 0, index: 8)
        globalEncoder.setBuffer(headBuffers["Wwdl"]!, offset: 0, index: 9)
        globalEncoder.setBuffer(headBuffers["bwdl"]!, offset: 0, index: 10)
        globalEncoder.setBuffer(headBuffers["Wscore"]!, offset: 0, index: 11)
        globalEncoder.setBuffer(headBuffers["bscore"]!, offset: 0, index: 12)
        globalEncoder.setBuffer(policyBuf, offset: 0, index: 13)
        globalEncoder.setBuffer(wdlBuf, offset: 0, index: 14)
        globalEncoder.setBuffer(scoreBuf, offset: 0, index: 15)
        globalEncoder.setBuffer(globalDimsBuf, offset: 0, index: 16)
        globalEncoder.dispatchThreads(MTLSize(width: B, height: 1, depth: 1), threadsPerThreadgroup: MetalPackedBackend.threadgroupSize1D(globalPipeline, n: B))
        globalEncoder.endEncoding()

        func readBack() throws -> RawBatch {
            let policyLogits = Array(UnsafeBufferPointer(start: policyBuf.contents().bindMemory(to: Float.self, capacity: B * (P + 1)), count: B * (P + 1)))
            let wdlLogits = Array(UnsafeBufferPointer(start: wdlBuf.contents().bindMemory(to: Float.self, capacity: B * 3), count: B * 3))
            let scoreMean = Array(UnsafeBufferPointer(start: scoreBuf.contents().bindMemory(to: Float.self, capacity: B), count: B))
            let ownership = Array(UnsafeBufferPointer(start: ownershipBuf.contents().bindMemory(to: Float.self, capacity: B * P), count: B * P))
            for (label, arr) in [("policyLogits", policyLogits), ("wdlLogits", wdlLogits), ("scoreMean", scoreMean), ("ownership", ownership)] {
                guard arr.allSatisfy({ $0.isFinite }) else { throw LogicModelError.nonFiniteOutput(label) }
            }
            return RawBatch(boardSize: S, batch: B, policyLogits: policyLogits, wdlLogits: wdlLogits, scoreMean: scoreMean, ownership: ownership)
        }
        return (readBack, keepAlive)
    }

    /// Output bits of every logic layer, unpacked to `[B,S,S,C]` `(((b*S+y)*S+x)*C+c)` order --
    /// bit-for-bit identical to `ScalarBackend.layerOutputs`/`MetalBackend.layerOutputs`. Runs
    /// only the gate dispatches (no heads) in their own command buffer, mirroring
    /// `PackedCPUBackend.layerOutputs`.
    public func layerOutputs(features: FeatureBatch) async throws -> [[UInt8]] {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let L = model.layers
        if B == 0 {
            return Array(repeating: [UInt8](), count: L)
        }
        let (commandBuffer, gateOutBuffers, G, gateKeepAlive) = try encodeGateLayers(features: features)
        try await MetalBackend.commitAndWait(commandBuffer)
        _ = gateKeepAlive // kept alive across the await above; nothing more to do with it here

        var result: [[UInt8]] = []
        result.reserveCapacity(L)
        let wordsPerLayer = G * S * S * C
        for l in 0 ..< L {
            let ptr = gateOutBuffers[l].contents().bindMemory(to: UInt32.self, capacity: wordsPerLayer)
            let packed = Array(UnsafeBufferPointer(start: ptr, count: wordsPerLayer))
            result.append(PackBits.unpack(packed, boardSize: S, batch: B, channels: C))
        }
        return result
    }

    /// Encodes the `L` packed gate-layer dispatches (into a fresh command buffer, not yet
    /// committed) and returns the buffers holding each layer's packed output, so callers can
    /// either commit immediately (`layerOutputs`) or append more encoders first (`evaluate`
    /// appends the head kernels before committing).
    private func encodeGateLayers(features: FeatureBatch) throws -> (commandBuffer: MTLCommandBuffer, outBuffers: [MTLBuffer], groups: Int, keepAlive: [MTLBuffer]) {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let L = model.layers
        let inputC = FeatureLayout.spatialChannels
        let G = PackBits.groupCount(batch: B)

        let inputPacked = PackBits.pack(features.spatial, boardSize: S, batch: B, channels: inputC)
        guard let inputBuffer = inputPacked.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }) else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal packed input buffer")
        }

        var outBuffers: [MTLBuffer] = []
        outBuffers.reserveCapacity(L)
        let layerWords = G * S * S * C
        for _ in 0 ..< L {
            guard let buf = device.makeBuffer(length: layerWords * 4, options: .storageModeShared) else {
                throw LogicModelError.backendUnavailable("cannot allocate Metal packed layer-output buffer")
            }
            outBuffers.append(buf)
        }

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw LogicModelError.backendUnavailable("cannot create Metal command buffer")
        }

        // `descriptorBuffers`/`inputBuffer` must stay alive until the caller's command buffer
        // completes; returned as `keepAlive` so the caller holds a strong reference across its
        // own `commitAndWait` (rather than capturing a mutable array in a `@Sendable` completion
        // handler, which Swift concurrency rightly flags for a non-Sendable `MTLBuffer`).
        var descriptorBuffers: [MTLBuffer] = [inputBuffer]
        descriptorBuffers.reserveCapacity(L * 3 + 1)

        for l in 0 ..< L {
            let wiringFlat = MetalPackedBackend.flattenWiring(model.wiring[l])
            let gatesForLayer = model.gates[l]
            let prevBuffer = l == 0 ? inputBuffer : outBuffers[l - 1]
            let prevC: Int32 = l == 0 ? Int32(inputC) : Int32(C)
            let dims: [Int32] = [Int32(G), Int32(S), Int32(C), Int32(inputC), prevC, Int32(B)]

            guard let wiringBuf = wiringFlat.withUnsafeBytes({ raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }), let gatesBuf = gatesForLayer.withUnsafeBytes({ raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }), let dimsBuf = dims.withUnsafeBytes({ raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }) else {
                throw LogicModelError.backendUnavailable("cannot allocate Metal descriptor buffer for layer \(l)")
            }
            descriptorBuffers.append(contentsOf: [wiringBuf, gatesBuf, dimsBuf])

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw LogicModelError.backendUnavailable("cannot create Metal compute encoder for layer \(l)")
            }
            encoder.setComputePipelineState(gatePipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(prevBuffer, offset: 0, index: 1)
            encoder.setBuffer(outBuffers[l], offset: 0, index: 2)
            encoder.setBuffer(wiringBuf, offset: 0, index: 3)
            encoder.setBuffer(gatesBuf, offset: 0, index: 4)
            encoder.setBuffer(dimsBuf, offset: 0, index: 5)
            let grid = MTLSize(width: S, height: S, depth: G * C)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: MetalPackedBackend.threadgroupSize3D(gatePipeline, s: S, z: G * C))
            encoder.endEncoding()
        }

        return (commandBuffer, outBuffers, G, descriptorBuffers)
    }

    /// `[C][2][4]` int32: (bank, channel, dx, dy) for reference A then B, per output channel.
    private static func flattenWiring(_ layer: [[GateReference]]) -> [Int32] {
        var out = [Int32]()
        out.reserveCapacity(layer.count * 8)
        for refs in layer {
            for r in refs {
                out.append(r.bank)
                out.append(r.channel)
                out.append(r.dx)
                out.append(r.dy)
            }
        }
        return out
    }

    private static func threadgroupSize1D(_ pipeline: MTLComputePipelineState, n: Int) -> MTLSize {
        MTLSize(width: max(1, min(n, pipeline.maxTotalThreadsPerThreadgroup)), height: 1, depth: 1)
    }

    private static func threadgroupSize2D(_ pipeline: MTLComputePipelineState, w: Int, h: Int) -> MTLSize {
        let tw = max(1, min(w, 32))
        let remaining = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        let th = max(1, min(h, remaining))
        return MTLSize(width: tw, height: th, depth: 1)
    }

    private static func threadgroupSize3D(_ pipeline: MTLComputePipelineState, s: Int, z: Int) -> MTLSize {
        let w = max(1, min(s, 8))
        let h = max(1, min(s, 8))
        let remaining = max(1, pipeline.maxTotalThreadsPerThreadgroup / (w * h))
        let d = max(1, min(z, remaining))
        return MTLSize(width: w, height: h, depth: d)
    }

    private static func kernelSource() throws -> String {
        func read(_ resource: String) throws -> String {
            guard let url = Bundle.module.url(forResource: resource, withExtension: "metal") else {
                throw LogicModelError.backendUnavailable("\(resource).metal resource not found in the LogicMetal bundle")
            }
            do {
                return try String(contentsOf: url, encoding: .utf8)
            } catch {
                throw LogicModelError.backendUnavailable("cannot read \(resource).metal: \(error)")
            }
        }
        return try read("logic_packed") + "\n" + read("heads")
    }
}
