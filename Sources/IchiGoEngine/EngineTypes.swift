/// Engine-level API types (docs/spec/03-engine.md §2).
import Foundation
import LogicModel

public struct ModelCapabilities: Sendable {
    public let boardSizes: Set<Int>
    public let rulesID: String
    public let hasOwnership: Bool
    /// v1 models never estimate score uncertainty; search features that rely on it must stay off.
    public let hasScoreUncertainty: Bool

    public init(boardSizes: Set<Int>, rulesID: String, hasOwnership: Bool, hasScoreUncertainty: Bool = false) {
        self.boardSizes = boardSizes
        self.rulesID = rulesID
        self.hasOwnership = hasOwnership
        self.hasScoreUncertainty = hasScoreUncertainty
    }
}

// Re-exported so search code can name the to-move evaluation without importing LogicModel.
public typealias LogicEvaluation = LogicModel.LogicEvaluation
