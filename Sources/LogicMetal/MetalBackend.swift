import Foundation
import LogicModel
import Metal

/// Metal-byte backend (docs/spec/01-network.md §5, docs/spec/04-tasks.md T22). Same `UInt8`
/// `[B,S,S,C]` activation layout as `ScalarBackend`. Every logic layer is one compute dispatch;
/// all `L` dispatches are encoded into a single `MTLCommandBuffer` with no CPU synchronisation in
/// between (Metal's default per-resource hazard tracking serialises the ping-pong reads/writes
/// across encoders). The command buffer is committed once and awaited once; only then are the
/// results read back and handed to the existing CPU `Heads.evaluate` (Metal heads are T24, out of
/// scope here). A command-buffer error throws before anything is read back — never partial
/// results.
///
/// `.metal` compilation: SwiftPM tools 6.0's own build system (`swift build`/`swift test`, as
/// opposed to Xcode's build system) does not compile `.metal` resources into a `.metallib`; a
/// `.process`/`.copy` resource rule just copies the source file into the resource bundle
/// verbatim. So `Resources/logic_byte.metal` ships as a bundled text resource and is compiled at
/// runtime via `MTLDevice.makeLibrary(source:options:)` the first time a `MetalBackend` is
/// constructed.
public final class MetalBackend: LogicBackend, @unchecked Sendable {
    public let model: LogicModelData
    public let name = "metal-byte"

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    /// Bytes currently allocated on the Metal device by this backend's buffers (best-effort,
    /// for benchmark reporting; docs/spec/05-validation.md §7 `metal_allocated_bytes`).
    public var currentAllocatedBytes: UInt64 { UInt64(device.currentAllocatedSize) }
    public var deviceName: String { device.name }

    public init(model: LogicModelData) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LogicModelError.backendUnavailable("no Metal device on this host")
        }
        guard let queue = device.makeCommandQueue() else {
            throw LogicModelError.backendUnavailable("cannot create a Metal command queue")
        }
        let source = try MetalBackend.kernelSource()
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw LogicModelError.backendUnavailable("cannot compile logic_byte.metal: \(error)")
        }
        guard let function = library.makeFunction(name: "logic_layer") else {
            throw LogicModelError.backendUnavailable("logic_layer function missing from compiled Metal library")
        }
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw LogicModelError.backendUnavailable("cannot create Metal compute pipeline: \(error)")
        }
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
        let layers = try await layerOutputs(features: features)
        // `Heads.evaluate` (the explicit-loop golden oracle) stays reserved for `ScalarBackend`;
        // every other backend, this one included, uses the faster reassociated-sum path
        // (docs/spec/04-tasks.md T24: "CPU Heads.evaluateを...高速化"), which
        // `HeadsAcceleratedTests` checks stays within docs/spec/05-validation.md §3 tolerance.
        return try Heads.evaluateAccelerated(model: model, lastLayer: layers[model.layers - 1], features: features)
    }

    /// Output bits of every logic layer, each `[B,S,S,C]` in `(((b*S+y)*S+x)*C+c)` order.
    /// Exposed for per-layer parity tests, mirroring `ScalarBackend.layerOutputs`. Unlike the CPU
    /// backend this must be `async throws`: buffer allocation or the command buffer itself can
    /// fail, and that failure must surface as a thrown error, never a partial/garbage result.
    public func layerOutputs(features: FeatureBatch) async throws -> [[UInt8]] {
        let S = features.boardSize
        let B = features.batch
        let C = model.channels
        let L = model.layers
        let inputC = FeatureLayout.spatialChannels
        if B == 0 {
            return Array(repeating: [UInt8](), count: L)
        }

        guard let inputBuffer = features.spatial.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }) else {
            throw LogicModelError.backendUnavailable("cannot allocate Metal input buffer")
        }

        var outBuffers: [MTLBuffer] = []
        outBuffers.reserveCapacity(L)
        let layerBytes = B * S * S * C
        for _ in 0 ..< L {
            guard let buf = device.makeBuffer(length: layerBytes, options: .storageModeShared) else {
                throw LogicModelError.backendUnavailable("cannot allocate Metal layer-output buffer")
            }
            outBuffers.append(buf)
        }

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw LogicModelError.backendUnavailable("cannot create Metal command buffer")
        }

        // Keep the small per-layer descriptor buffers alive until the command buffer completes.
        var descriptorBuffers: [MTLBuffer] = []
        descriptorBuffers.reserveCapacity(L * 3)

        for l in 0 ..< L {
            let wiringFlat = MetalBackend.flattenWiring(model.wiring[l])
            let gatesForLayer = model.gates[l]
            let prevBuffer = l == 0 ? inputBuffer : outBuffers[l - 1]
            let prevC: Int32 = l == 0 ? Int32(inputC) : Int32(C)
            let dims: [Int32] = [Int32(B), Int32(S), Int32(C), Int32(inputC), prevC]

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
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(prevBuffer, offset: 0, index: 1)
            encoder.setBuffer(outBuffers[l], offset: 0, index: 2)
            encoder.setBuffer(wiringBuf, offset: 0, index: 3)
            encoder.setBuffer(gatesBuf, offset: 0, index: 4)
            encoder.setBuffer(dimsBuf, offset: 0, index: 5)
            let grid = MTLSize(width: S, height: S, depth: B * C)
            encoder.dispatchThreads(grid, threadsPerThreadgroup: MetalBackend.threadgroupSize(pipeline: pipeline, s: S, z: B * C))
            encoder.endEncoding()
        }

        // One commit, one wait: no CPU synchronisation between layers.
        try await MetalBackend.commitAndWait(commandBuffer)

        var result: [[UInt8]] = []
        result.reserveCapacity(L)
        for l in 0 ..< L {
            let ptr = outBuffers[l].contents().bindMemory(to: UInt8.self, capacity: layerBytes)
            result.append(Array(UnsafeBufferPointer(start: ptr, count: layerBytes)))
        }
        return result
    }

    /// Not `private`: `MetalPackedBackend` (same module) reuses this exact commit/await/error
    /// shape for its own command buffers.
    static func commitAndWait(_ commandBuffer: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buf in
                if let error = buf.error {
                    continuation.resume(throwing: LogicModelError.backendUnavailable("Metal command buffer failed: \(error)"))
                } else if buf.status != .completed {
                    continuation.resume(throwing: LogicModelError.backendUnavailable("Metal command buffer ended in status \(buf.status.rawValue), not completed"))
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }
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

    /// A conservative threadgroup shape for the (x, y, b*C+c) grid: caps x/y at 8 (S is at most
    /// 19) and fills the remaining threadgroup budget along z (the batch*channel axis, which is
    /// usually the largest). `dispatchThreads` uses non-uniform threadgroups, so grid sizes that
    /// are not multiples of the threadgroup size are handled correctly by Metal.
    private static func threadgroupSize(pipeline: MTLComputePipelineState, s: Int, z: Int) -> MTLSize {
        let w = max(1, min(s, 8))
        let h = max(1, min(s, 8))
        let remaining = max(1, pipeline.maxTotalThreadsPerThreadgroup / (w * h))
        let d = max(1, min(z, remaining))
        return MTLSize(width: w, height: h, depth: d)
    }

    private static func kernelSource() throws -> String {
        guard let url = Bundle.module.url(forResource: "logic_byte", withExtension: "metal") else {
            throw LogicModelError.backendUnavailable("logic_byte.metal resource not found in the LogicMetal bundle")
        }
        do {
            return try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw LogicModelError.backendUnavailable("cannot read logic_byte.metal: \(error)")
        }
    }
}
