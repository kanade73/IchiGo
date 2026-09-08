import Foundation

/// Errors raised by LogicModel. Every failure is explicit; no partial results are returned.
public enum LogicModelError: Error, CustomStringConvertible, Equatable {
    case invalidManifest(String)
    case invalidPayload(String)
    case invalidInput(String)
    case nonFiniteOutput(String)
    case backendUnavailable(String)

    public var description: String {
        switch self {
        case let .invalidManifest(m): "invalid manifest: \(m)"
        case let .invalidPayload(m): "invalid payload: \(m)"
        case let .invalidInput(m): "invalid input: \(m)"
        case let .nonFiniteOutput(m): "non-finite output: \(m)"
        case let .backendUnavailable(m): "backend unavailable: \(m)"
        }
    }
}

public enum FeatureLayout {
    /// Spatial input channels (docs/spec/01-network.md §1).
    public static let spatialChannels = 32
    /// Global input features.
    public static let globalFeatures = 4
    public static let featureVersion = 1
}

/// A batch of encoded positions, all of the same board size.
/// - `spatial`: `[B,S,S,32] uint8`, index `(((b*S+y)*S+x)*32+c)`, values exactly 0/1.
/// - `global`: `[B,4] float32`.
/// - `legal`: `[B,S*S+1] uint8`, pass at index `S*S`.
public struct FeatureBatch: Sendable, Equatable {
    public let boardSize: Int
    public let batch: Int
    public let spatial: [UInt8]
    public let global: [Float]
    public let legal: [UInt8]

    public init(boardSize: Int, batch: Int, spatial: [UInt8], global: [Float], legal: [UInt8]) throws {
        guard boardSize >= 1, boardSize <= 19 else { throw LogicModelError.invalidInput("boardSize \(boardSize) out of range") }
        guard batch >= 0 else { throw LogicModelError.invalidInput("negative batch") }
        let s = boardSize
        guard spatial.count == batch * s * s * FeatureLayout.spatialChannels else {
            throw LogicModelError.invalidInput("spatial count \(spatial.count) != B*S*S*32 for B=\(batch), S=\(s)")
        }
        guard global.count == batch * FeatureLayout.globalFeatures else {
            throw LogicModelError.invalidInput("global count \(global.count) != B*4")
        }
        guard legal.count == batch * (s * s + 1) else {
            throw LogicModelError.invalidInput("legal count \(legal.count) != B*(S*S+1)")
        }
        guard spatial.allSatisfy({ $0 <= 1 }) else { throw LogicModelError.invalidInput("spatial values must be 0/1") }
        guard legal.allSatisfy({ $0 <= 1 }) else { throw LogicModelError.invalidInput("legal values must be 0/1") }
        guard global.allSatisfy({ $0.isFinite }) else { throw LogicModelError.invalidInput("global contains non-finite values") }
        self.boardSize = boardSize
        self.batch = batch
        self.spatial = spatial
        self.global = global
        self.legal = legal
    }
}

/// Raw head outputs, to-move perspective, same order as the input batch.
/// - `policyLogits`: `[B,S*S+1]` (board points `y*S+x`, then pass), unmasked logits.
/// - `wdlLogits`: `[B,3]` (win, draw, loss). `scoreMean`: `[B]` points incl. komi.
/// - `ownership`: `[B,S*S]` in [-1,1] (+1 = to-move owns).
public struct RawBatch: Sendable, Equatable {
    public let boardSize: Int
    public let batch: Int
    public let policyLogits: [Float]
    public let wdlLogits: [Float]
    public let scoreMean: [Float]
    public let ownership: [Float]

    public init(boardSize: Int, batch: Int, policyLogits: [Float], wdlLogits: [Float], scoreMean: [Float], ownership: [Float]) {
        self.boardSize = boardSize
        self.batch = batch
        self.policyLogits = policyLogits
        self.wdlLogits = wdlLogits
        self.scoreMean = scoreMean
        self.ownership = ownership
    }
}

/// Backend contract (docs/spec/01-network.md §5). B=0 returns an empty batch; mixed sizes are
/// impossible by construction of `FeatureBatch`; non-finite outputs throw.
public protocol LogicBackend: Sendable {
    var name: String { get }
    func evaluate(features: FeatureBatch) async throws -> RawBatch
}
