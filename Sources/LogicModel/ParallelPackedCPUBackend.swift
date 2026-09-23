import Foundation

/// `cpu-packed` spread over several cores: the batch is cut into contiguous chunks of at least
/// `minChunk` positions, each chunk is evaluated by `PackedCPUBackend.evaluateSync` in its own
/// child task, and the outputs are concatenated in order. Positions never interact inside a
/// backend, so the logic layers are bit-identical to one `PackedCPUBackend` over the whole batch;
/// head floats can differ in the last bits (Accelerate blocks the head matmul differently for a
/// different batch size), within the same tolerance as packed vs scalar.
///
/// Measured on an M5 (4 performance + 6 efficiency cores, wide512 model): one core does ~7.7k
/// positions/s at batch 32, ten concurrent single-core workers ~41k (2026-09-23), so throughput
/// scales with cores as long as each chunk still fills most of a 32-lane packed word.
public struct ParallelPackedCPUBackend: LogicBackend {
    public let name = "cpu-packed-mt"
    public let inner: PackedCPUBackend
    public let workers: Int
    public let minChunk: Int

    public init(model: LogicModelData, workers: Int = ProcessInfo.processInfo.activeProcessorCount, minChunk: Int = 32) {
        inner = PackedCPUBackend(model: model)
        self.workers = max(1, workers)
        self.minChunk = max(1, minChunk)
    }

    public func evaluate(features: FeatureBatch) async throws -> RawBatch {
        let B = features.batch
        let chunks = max(1, min(workers, (B + minChunk - 1) / minChunk))
        if chunks == 1 { return try inner.evaluateSync(features: features) }
        let per = (B + chunks - 1) / chunks
        let ranges = stride(from: 0, to: B, by: per).map { $0 ..< min(B, $0 + per) }
        let inner = inner
        return try await withThrowingTaskGroup(of: (Int, RawBatch).self) { group in
            for (i, r) in ranges.enumerated() {
                let part = features.slice(r)
                group.addTask { (i, try inner.evaluateSync(features: part)) }
            }
            var parts = [RawBatch?](repeating: nil, count: ranges.count)
            for try await (i, raw) in group { parts[i] = raw }
            return RawBatch.concatenating(parts.map { $0! }, boardSize: features.boardSize)
        }
    }
}

extension FeatureBatch {
    /// Positions `range` of this batch. The values were validated when this batch was built, so
    /// the slice skips re-validation.
    func slice(_ range: Range<Int>) -> FeatureBatch {
        let S = boardSize, C = FeatureLayout.spatialChannels, G = FeatureLayout.globalFeatures, P = S * S + 1
        return FeatureBatch(
            unchecked: S, batch: range.count,
            spatial: Array(spatial[(range.lowerBound * S * S * C) ..< (range.upperBound * S * S * C)]),
            global: Array(global[(range.lowerBound * G) ..< (range.upperBound * G)]),
            legal: Array(legal[(range.lowerBound * P) ..< (range.upperBound * P)])
        )
    }
}

extension RawBatch {
    /// Concatenates per-chunk outputs in order (the inverse of `FeatureBatch.slice`).
    static func concatenating(_ parts: [RawBatch], boardSize: Int) -> RawBatch {
        RawBatch(
            boardSize: boardSize, batch: parts.reduce(0) { $0 + $1.batch },
            policyLogits: parts.flatMap(\.policyLogits), wdlLogits: parts.flatMap(\.wdlLogits),
            scoreMean: parts.flatMap(\.scoreMean), ownership: parts.flatMap(\.ownership)
        )
    }
}
