import Foundation
import IchiGoEngine

/// Parses the `--value-source`/`--value-blend`/`--value-k`/`--value-b` CLI flags shared by
/// `ichigo gtp` and `ichigo selfplay` (docs/spec/03-engine.md §3-4 "値ソース") into a
/// `ValueSource`. Kept here (rather than inline in `Sources/ichigo/main.swift`) so the parsing
/// logic is unit-testable without an `ichigo` test target.
public enum ValueSourceFlagError: Error, Equatable, CustomStringConvertible {
    case unknownSource(String)

    public var description: String {
        switch self {
        case let .unknownSource(s): "--value-source must be network, ownership, or blend, got \(s)"
        }
    }
}

public enum ValueSourceFlags {
    /// `source` defaults to `"network"` when absent; `blend`/`k`/`b` fall back to their documented
    /// defaults (0.5, 6, 1.0) when absent or unparsable, matching how other numeric CLI flags in
    /// `ichigo` already behave (e.g. `--visits`, `--warmup`). `source` itself is validated
    /// strictly, like `--backend`.
    public static func parse(source: String?, blend: String?, k: String?, b: String?) throws -> ValueSource {
        let kf = k.flatMap(Float.init) ?? 6
        let bf = b.flatMap(Float.init) ?? 1.0
        let wf = blend.flatMap(Float.init) ?? 0.5
        switch source ?? "network" {
        case "network": return .network
        case "ownership": return .ownership(k: kf, b: bf)
        case "blend": return .blend(weightNetwork: wf, k: kf, b: bf)
        default: throw ValueSourceFlagError.unknownSource(source ?? "")
        }
    }
}
