import Foundation
import LogicModel

/// A saved "which backend was fastest at this batch size" table (docs/spec/04-tasks.md T25:
/// "batch別backend選択ファイル生成"). `ichigo benchmark` writes one of these next to its
/// `--out` report; `configs/backend-profile.example.json` documents the on-disk shape. This is
/// deliberately *not* a raw measurement dump: it stores only the already-decided winner per
/// batch size, so a caller does not need to re-implement the tie-breaking rule.
public struct BackendProfile: Sendable, Equatable {
    public var modelHash: String?
    public var hardware: String?
    /// batch size -> backend name ("cpu" or "metal").
    public var batches: [Int: String]

    public init(modelHash: String? = nil, hardware: String? = nil, batches: [Int: String] = [:]) {
        self.modelHash = modelHash
        self.hardware = hardware
        self.batches = batches
    }

    public enum LoadError: Error, CustomStringConvertible {
        case notFound(String)
        case invalidJSON(String)

        public var description: String {
            switch self {
            case let .notFound(p): "backend profile not found at \(p)"
            case let .invalidJSON(m): "invalid backend profile JSON: \(m)"
            }
        }
    }

    public static func load(path: String) throws -> BackendProfile {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw LoadError.notFound(path)
        }
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw LoadError.invalidJSON("top level is not a JSON object")
        }
        guard let batchesAny = obj["batches"] as? [String: Any] else {
            throw LoadError.invalidJSON("missing \"batches\" object")
        }
        var batches: [Int: String] = [:]
        for (key, value) in batchesAny {
            guard let b = Int(key), let backend = value as? String else {
                throw LoadError.invalidJSON("batches.\(key) must map an integer-string key to a backend name string")
            }
            batches[b] = backend
        }
        return BackendProfile(modelHash: obj["modelHash"] as? String, hardware: obj["hardware"] as? String, batches: batches)
    }

    /// Serialisable form matching `configs/backend-profile.example.json`.
    public func toJSONObject() -> [String: Any] {
        var obj: [String: Any] = [
            "format": "ichigo.backend-profile", "version": 1,
            "batches": Dictionary(uniqueKeysWithValues: batches.map { (String($0.key), $0.value) }),
        ]
        if let modelHash { obj["modelHash"] = modelHash }
        if let hardware { obj["hardware"] = hardware }
        return obj
    }
}

/// Picks a backend name ("cpu" or "metal") for a given batch size (docs/spec/04-tasks.md T25).
/// With no profile, or no entry for the requested batch, this is the same "auto" rule as
/// `ichigo eval/gtp --backend auto`: Metal if a device is available, otherwise CPU
/// (docs/spec/01-network.md §5: "Metal device がなければ...auto はCPU").
public enum BackendSelector {
    public static let cpu = "cpu"
    public static let metal = "metal"

    public static func select(batch: Int, profile: BackendProfile?, metalAvailable: Bool) -> String {
        if let picked = profile?.batches[batch] {
            // A saved profile is only trustworthy on the machine it was measured on; if it names
            // a backend that is not actually available here, fall back to auto rather than
            // handing the caller a backend that will throw.
            if picked == metal, !metalAvailable { return cpu }
            return picked
        }
        return metalAvailable ? metal : cpu
    }

    /// Reduces a set of (batch, backend, positionsPerSec) measurements to the fastest backend per
    /// batch size, i.e. the "手順: batch別backend選択ファイル生成" step of T25.
    public static func fastestPerBatch(measurements: [(batch: Int, backend: String, positionsPerSec: Double)]) -> [Int: String] {
        var best: [Int: (backend: String, rate: Double)] = [:]
        for m in measurements {
            if let current = best[m.batch] {
                if m.positionsPerSec > current.rate { best[m.batch] = (m.backend, m.positionsPerSec) }
            } else {
                best[m.batch] = (m.backend, m.positionsPerSec)
            }
        }
        return best.mapValues(\.backend)
    }
}
