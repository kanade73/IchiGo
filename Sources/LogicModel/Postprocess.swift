import Foundation

/// Post-processed evaluation of one position (docs/spec/03-engine.md §2–3), to-move perspective.
/// - `policy`: `[S*S+1]`, legal moves sum to 1, illegal moves exactly 0.
/// - `winDrawLoss`: `[3]` softmax of wdl logits. `expectedResult = win + draw/2`.
/// - `scoreMean`: points, to-move perspective, komi included. `ownership`: `[S*S]` in [-1,1].
public struct LogicEvaluation: Sendable, Equatable {
    public let policy: [Float]
    public let winDrawLoss: [Float]
    public let expectedResult: Float
    public let scoreMean: Float
    public let ownership: [Float]

    public init(policy: [Float], winDrawLoss: [Float], expectedResult: Float, scoreMean: Float, ownership: [Float]) {
        self.policy = policy
        self.winDrawLoss = winDrawLoss
        self.expectedResult = expectedResult
        self.scoreMean = scoreMean
        self.ownership = ownership
    }
}

public enum Postprocess {
    /// Stable masked softmax. Throws when a position has no legal move, or a value is non-finite.
    public static func evaluate(raw: RawBatch, features: FeatureBatch) throws -> [LogicEvaluation] {
        guard raw.batch == features.batch, raw.boardSize == features.boardSize else {
            throw LogicModelError.invalidInput("raw/features batch or size mismatch")
        }
        let P = raw.boardSize * raw.boardSize
        var out: [LogicEvaluation] = []
        for b in 0 ..< raw.batch {
            let logits = raw.policyLogits[(b * (P + 1)) ..< ((b + 1) * (P + 1))]
            let legal = features.legal[(b * (P + 1)) ..< ((b + 1) * (P + 1))]
            var maxLogit = -Float.infinity
            for (l, m) in zip(logits, legal) where m == 1 { maxLogit = max(maxLogit, l) }
            guard maxLogit.isFinite else { throw LogicModelError.invalidInput("position \(b) has no legal move") }
            var policy = [Float](repeating: 0, count: P + 1)
            var sum: Double = 0
            for (i, (l, m)) in zip(logits, legal).enumerated() where m == 1 {
                let e = exp(Double(l - maxLogit))
                policy[i] = Float(e)
                sum += e
            }
            for i in 0 ..< (P + 1) where policy[i] != 0 { policy[i] = Float(Double(policy[i]) / sum) }
            let w = Array(raw.wdlLogits[(b * 3) ..< (b * 3 + 3)])
            let wm = w.max()!
            let we = w.map { exp(Double($0 - wm)) }
            let ws = we.reduce(0, +)
            let wdl = we.map { Float($0 / ws) }
            let expected = wdl[0] + 0.5 * wdl[1]
            let own = Array(raw.ownership[(b * P) ..< ((b + 1) * P)])
            guard policy.allSatisfy({ $0.isFinite }), wdl.allSatisfy({ $0.isFinite }), raw.scoreMean[b].isFinite else {
                throw LogicModelError.nonFiniteOutput("postprocess position \(b)")
            }
            out.append(LogicEvaluation(policy: policy, winDrawLoss: wdl, expectedResult: expected, scoreMean: raw.scoreMean[b], ownership: own))
        }
        return out
    }
}
