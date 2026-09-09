import Foundation
import IchiGoEngine

/// Parses the `--value-source`/`--value-blend`/`--value-k`/`--value-b`/`--rollout-count`/
/// `--rollout-max-moves` CLI flags shared by `ichigo gtp` and `ichigo selfplay`
/// (docs/spec/03-engine.md §3-4 "値ソース") into a `ValueSource`. Kept here (rather than inline in
/// `Sources/ichigo/main.swift`) so the parsing logic is unit-testable without an `ichigo` test
/// target.
public enum ValueSourceFlagError: Error, Equatable, CustomStringConvertible {
    case unknownSource(String)

    public var description: String {
        switch self {
        case let .unknownSource(s): "--value-source must be network, ownership, blend, or rollout, got \(s)"
        }
    }
}

public enum ValueSourceFlags {
    /// `source` defaults to `"network"` when absent; `blend`/`k`/`b`/`rolloutCount` fall back to
    /// their documented defaults (0.5, 6, 1.0, 8) when absent or unparsable, matching how other
    /// numeric CLI flags in `ichigo` already behave (e.g. `--visits`, `--warmup`). `source` itself
    /// is validated strictly, like `--backend`. `rolloutMaxMoves` defaults to `2*boardSize^2`
    /// (docs/implementation-status.md 2026-09-10 §4-5) when absent or unparsable — `boardSize` is
    /// the caller's active board size (`gtp`'s `--model-9`/`--model-19`/`--model` selection,
    /// `selfplay`'s `--size`), since `rollout` is the only mode whose default depends on it.
    public static func parse(
        source: String?, blend: String?, k: String?, b: String?,
        rolloutCount: String? = nil, rolloutMaxMoves: String? = nil, boardSize: Int = 9
    ) throws -> ValueSource {
        let kf = k.flatMap(Float.init) ?? 6
        let bf = b.flatMap(Float.init) ?? 1.0
        let wf = blend.flatMap(Float.init) ?? 0.5
        switch source ?? "network" {
        case "network": return .network
        case "ownership": return .ownership(k: kf, b: bf)
        case "blend": return .blend(weightNetwork: wf, k: kf, b: bf)
        case "rollout":
            let count = rolloutCount.flatMap(Int.init) ?? 8
            let maxMoves = rolloutMaxMoves.flatMap(Int.init) ?? (2 * boardSize * boardSize)
            return .rollout(count: count, maxMoves: maxMoves, weightNetwork: wf)
        default: throw ValueSourceFlagError.unknownSource(source ?? "")
        }
    }
}
