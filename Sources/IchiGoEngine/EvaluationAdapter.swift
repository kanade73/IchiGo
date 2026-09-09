import Foundation
import IchiGoCore
import IchiGoFeatures
import LogicModel

/// White-perspective view used by the ported RinGo search (docs/spec/03-engine.md §3).
/// `whiteWinProb` is the expected score (win + draw/2), NOT a strict win probability; the raw WDL
/// is kept separately. Score variance/error fields are unestimated in v1 and fixed to 0 — search
/// features that consume them must stay disabled (see `ModelCapabilities.hasScoreUncertainty`).
public struct WhiteEvaluation: Sendable, Equatable {
    /// Policy over `S*S+1` moves; illegal moves carry the RinGo sentinel `-1`.
    public let policy: [Float]
    public let whiteExpected: Float
    public let whiteWinValue: Float      // 2*whiteExpected - 1, in [-1, 1]
    public let whiteWinProb: Float       // == whiteExpected (compat)
    public let whiteLossProb: Float      // == 1 - whiteExpected (compat)
    public let whiteNoResultProb: Float  // always 0 (draw is never mapped to no-result)
    public let whiteScoreMean: Float
    public let whiteScoreMeanSq: Float   // whiteScoreMean^2 (compat; not an estimate of variance)
    public let whiteLead: Float          // == whiteScoreMean
    public let varTimeLeft: Float        // 0, unestimated
    public let shorttermWinlossError: Float  // 0, unestimated
    public let shorttermScoreError: Float    // 0, unestimated
    public let whiteOwnership: [Float]
    /// Raw to-move WDL (win, draw, loss) from the network. Always the network's own output,
    /// regardless of `ValueSource` — see `EvaluationAdapter.toWhite`.
    public let rawWinDrawLoss: [Float]
    public let toMove: Player
}

/// Alternative leaf-value source for the search (docs/spec/03-engine.md §3-4 "値ソース").
/// Diagnostics (docs/implementation-status.md 2026-09-10) found the logic network's wdl head weak
/// on 9x9, while a value derived from the ownership head reaches comparable expected-result MAE
/// with no additional training:
///
///   score_est = Σ_xy ownership_xy + komiSelf         (komiSelf = toMove == white ? komi : -komi)
///   value_own = sigmoid((score_est + b) / k)          (to-move perspective, in [0,1])
///
/// `.ownership` uses `value_own` in place of the network's `expectedResult`; `.blend` uses
/// `weightNetwork * e_nn + (1 - weightNetwork) * value_own`. `EvaluationAdapter.toWhite` always
/// keeps the network's raw WDL in `rawWinDrawLoss` (and callers keep `e.expectedResult`/
/// `LogicEvaluation` itself untouched) so `e_nn` stays observable regardless of mode — PUCT only
/// needs *some* consistent white-perspective expected score in [0,1] (docs/spec/03-engine.md §4),
/// so swapping its source is a value-type change, not a rule change.
public struct ValueSource: Sendable, Equatable {
    public enum Mode: Sendable, Equatable {
        case network
        /// `k`, `b` as in `value_own` above.
        case ownership(k: Float, b: Float)
        /// `value = weightNetwork * e_nn + (1 - weightNetwork) * value_own`, both terms in the
        /// to-move perspective before the white-perspective conversion.
        case blend(weightNetwork: Float, k: Float, b: Float)
        /// Classic policy-guided playouts (docs/implementation-status.md 2026-09-10 §4-5:
        /// "playouts as an alternative/supplement to the network value"). At leaf expansion, after
        /// the network evaluation, `count` playouts are run from the leaf: each copies the leaf's
        /// `GameState` and repeatedly samples a move from the network's own policy (temperature 1
        /// over legal moves) until two passes or `maxMoves` plies, scored exactly by
        /// `GameState.exactWhiteOutcome` (or, if `maxMoves` is hit first, by
        /// `history.endAndScoreGameNow` on the as-is board — an approximation, see
        /// `Search.RolloutDiagnostics`). `value_rollout` (white perspective) is the mean of
        /// win=1/draw=0.5/loss=0 over the `count` playouts; the leaf value is
        /// `weightNetwork * e_nn(white) + (1 - weightNetwork) * value_rollout`. This blend needs
        /// the async evaluator and a seeded RNG that this synchronous adapter doesn't have, so it
        /// is computed by `Search.applyRollout`, not here — see `toWhite` below.
        case rollout(count: Int, maxMoves: Int, weightNetwork: Float)
    }

    public var mode: Mode
    public init(mode: Mode = .network) { self.mode = mode }

    public static let network = ValueSource(mode: .network)
    public static func ownership(k: Float = 6, b: Float = 1) -> ValueSource { ValueSource(mode: .ownership(k: k, b: b)) }
    public static func blend(weightNetwork: Float = 0.5, k: Float = 6, b: Float = 1) -> ValueSource {
        ValueSource(mode: .blend(weightNetwork: weightNetwork, k: k, b: b))
    }
    public static func rollout(count: Int = 8, maxMoves: Int, weightNetwork: Float = 0.5) -> ValueSource {
        ValueSource(mode: .rollout(count: count, maxMoves: maxMoves, weightNetwork: weightNetwork))
    }

    /// `score_est = Σ_xy ownership_xy + komiSelf`, to-move perspective. `ownership` is
    /// `LogicEvaluation.ownership` (to-move perspective, +1 = to-move owns).
    public static func scoreEstimate(ownership: [Float], komi: Float, toMove: Player) -> Float {
        let komiSelf: Float = toMove == .white ? komi : -komi
        return ownership.reduce(0, +) + komiSelf
    }

    /// `value_own = sigmoid((score_est + b) / k)`, to-move perspective, in [0,1].
    public static func sigmoidValue(_ scoreEst: Float, k: Float, b: Float) -> Float {
        Float(1 / (1 + exp(-(Double(scoreEst) + Double(b)) / Double(k))))
    }

    /// Compact one-line description used by GTP startup/genmove diagnostics logs.
    public var logDescription: String {
        switch mode {
        case .network: return "network"
        case let .ownership(k, b): return "ownership(k=\(k),b=\(b))"
        case let .blend(w, k, b): return "blend(w=\(w),k=\(k),b=\(b))"
        case let .rollout(count, maxMoves, w): return "rollout(count=\(count),maxMoves=\(maxMoves),w=\(w))"
        }
    }
}

public enum EvaluationAdapter {
    public static let illegalPolicySentinel: Float = -1

    /// Converts a to-move evaluation into the white-perspective compat view. `komi` is only
    /// consumed by `valueSource` modes other than `.network` (`score_est` needs it); it defaults
    /// to 0 so existing call sites that only ever want `.network` are unaffected.
    public static func toWhite(_ e: LogicEvaluation, toMove: Player, legal: [UInt8], komi: Float = 0, valueSource: ValueSource = .network) -> WhiteEvaluation {
        precondition(legal.count == e.policy.count, "legal mask and policy must have the same length")
        let sign: Float = toMove == .white ? 1 : -1
        var policy = e.policy
        for i in policy.indices where legal[i] == 0 { policy[i] = illegalPolicySentinel }

        // `toMoveExpected`/`toMoveScore` are the to-move-perspective value/score-lead that search
        // actually backs up. `e.expectedResult`/`e.winDrawLoss` (the network's raw output) are
        // never mutated by this — `rawWinDrawLoss` below always carries them unchanged, so e_nn
        // stays observable in logs/diagnostics no matter which value source is configured.
        let toMoveExpected: Float
        let toMoveScore: Float
        switch valueSource.mode {
        case .network:
            toMoveExpected = e.expectedResult
            toMoveScore = e.scoreMean
        case let .ownership(k, b):
            let scoreEst = ValueSource.scoreEstimate(ownership: e.ownership, komi: komi, toMove: toMove)
            toMoveExpected = ValueSource.sigmoidValue(scoreEst, k: k, b: b)
            toMoveScore = scoreEst
        case let .blend(w, k, b):
            let scoreEst = ValueSource.scoreEstimate(ownership: e.ownership, komi: komi, toMove: toMove)
            let ownExpected = ValueSource.sigmoidValue(scoreEst, k: k, b: b)
            toMoveExpected = w * e.expectedResult + (1 - w) * ownExpected
            // Score lead follows the ownership estimate whenever mode != .network, regardless of
            // the blend weight (docs/spec/03-engine.md §3-4): only the value/expected-result term
            // is actually blended between e_nn and value_own.
            toMoveScore = scoreEst
        case .rollout:
            // `Search.expand` never actually calls this adapter with `.rollout` (it substitutes
            // `.network` for the pre-rollout baseline, then `Search.applyRollout` overwrites the
            // node's white value after the async playouts finish — see the case's doc comment
            // above). This arm exists purely so the switch stays exhaustive for any other caller;
            // it behaves exactly like `.network`.
            toMoveExpected = e.expectedResult
            toMoveScore = e.scoreMean
        }

        let whiteExpected = toMove == .white ? toMoveExpected : 1 - toMoveExpected
        let score = sign * toMoveScore
        return WhiteEvaluation(
            policy: policy,
            whiteExpected: whiteExpected,
            whiteWinValue: 2 * whiteExpected - 1,
            whiteWinProb: whiteExpected,
            whiteLossProb: 1 - whiteExpected,
            whiteNoResultProb: 0,
            whiteScoreMean: score,
            whiteScoreMeanSq: score * score,
            whiteLead: score,
            varTimeLeft: 0,
            shorttermWinlossError: 0,
            shorttermScoreError: 0,
            whiteOwnership: e.ownership.map { sign * $0 },
            rawWinDrawLoss: e.winDrawLoss,
            toMove: toMove
        )
    }

    public static func toWhite(_ e: LogicEvaluation, snapshot: PositionSnapshot, valueSource: ValueSource = .network) -> WhiteEvaluation {
        toWhite(e, toMove: snapshot.toMove, legal: snapshot.legal, komi: snapshot.komi, valueSource: valueSource)
    }
}
