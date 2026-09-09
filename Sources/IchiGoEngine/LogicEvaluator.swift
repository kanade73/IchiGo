import Foundation
import IchiGoFeatures
import LogicModel

/// Position-level evaluation contract (docs/spec/03-engine.md §2). Results are to-move perspective.
public protocol PositionEvaluating: Actor {
    var capabilities: ModelCapabilities { get }
    /// Evaluates snapshots in order. Throws on an unsupported size, a finished game (exact
    /// outcomes are the caller's job), a mixed-size batch, or a backend failure. Never returns
    /// partial results.
    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation]
    func preWarm(size: Int) async throws
}

public enum EvaluatorError: Error, Equatable, CustomStringConvertible {
    case unsupportedBoardSize(Int)
    case mixedBoardSizes
    case gameFinished(index: Int)
    case countMismatch

    public var description: String {
        switch self {
        case let .unsupportedBoardSize(s): "board size \(s) not supported by the loaded model"
        case .mixedBoardSizes: "a batch must contain a single board size"
        case let .gameFinished(i): "position \(i) is finished on the board; evaluate exact outcome instead"
        case .countMismatch: "backend returned a different number of results"
        }
    }
}

/// Encodes snapshots (docs/spec/01-network.md §1), runs a `LogicBackend`, applies the masked
/// softmax and returns to-move `LogicEvaluation`s. Snapshots are value types, so the actor
/// boundary carries plain arrays only.
public actor LogicEvaluator: PositionEvaluating {
    public let capabilities: ModelCapabilities
    public let modelHash: String
    private let backend: any LogicBackend
    /// docs/spec/03-engine.md §9: divides the wdl logits before softmax so search sees the same
    /// calibrated evaluation as `ichigo eval`/analysis. From `model.manifest.calibrationTemperature`.
    private let temperature: Float

    public init(model: LogicModelData, backend: any LogicBackend) {
        capabilities = ModelCapabilities(boardSizes: Set(model.manifest.boardSizes), rulesID: ModelManifest.rulesID, hasOwnership: true)
        modelHash = model.payloadHash
        self.backend = backend
        temperature = model.manifest.calibrationTemperature
    }

    /// Test/alternate-backend initialiser (fake evaluators live in Tests only).
    public init(capabilities: ModelCapabilities, modelHash: String, backend: any LogicBackend, temperature: Float = 1.0) {
        self.capabilities = capabilities
        self.modelHash = modelHash
        self.backend = backend
        self.temperature = temperature
    }

    public func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        guard let first = positions.first else { return [] }
        guard capabilities.boardSizes.contains(first.boardSize) else { throw EvaluatorError.unsupportedBoardSize(first.boardSize) }
        guard positions.allSatisfy({ $0.boardSize == first.boardSize }) else { throw EvaluatorError.mixedBoardSizes }
        if let i = positions.firstIndex(where: { $0.isGameFinished }) { throw EvaluatorError.gameFinished(index: i) }
        let enc = try FeatureEncoder.encode(positions)
        let features = try FeatureBatch(boardSize: enc.boardSize, batch: enc.batch, spatial: enc.spatial, global: enc.global, legal: enc.legal)
        let raw = try await backend.evaluate(features: features)
        let out = try Postprocess.evaluate(raw: raw, features: features, temperature: temperature)
        guard out.count == positions.count else { throw EvaluatorError.countMismatch }
        return out
    }

    public func preWarm(size: Int) async throws {
        guard capabilities.boardSizes.contains(size) else { throw EvaluatorError.unsupportedBoardSize(size) }
        let g = try GameState(boardSize: size, komi: size == 9 ? IchiGoRules.defaultKomi9 : IchiGoRules.defaultKomi19)
        _ = try await evaluate([g.snapshot()])
    }
}
