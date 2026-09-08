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
    /// Raw to-move WDL (win, draw, loss) from the network.
    public let rawWinDrawLoss: [Float]
    public let toMove: Player
}

public enum EvaluationAdapter {
    public static let illegalPolicySentinel: Float = -1

    /// Converts a to-move evaluation into the white-perspective compat view.
    public static func toWhite(_ e: LogicEvaluation, toMove: Player, legal: [UInt8]) -> WhiteEvaluation {
        precondition(legal.count == e.policy.count, "legal mask and policy must have the same length")
        let sign: Float = toMove == .white ? 1 : -1
        let whiteExpected = toMove == .white ? e.expectedResult : 1 - e.expectedResult
        var policy = e.policy
        for i in policy.indices where legal[i] == 0 { policy[i] = illegalPolicySentinel }
        let score = sign * e.scoreMean
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

    public static func toWhite(_ e: LogicEvaluation, snapshot: PositionSnapshot) -> WhiteEvaluation {
        toWhite(e, toMove: snapshot.toMove, legal: snapshot.legal)
    }
}
