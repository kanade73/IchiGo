import Foundation
import LogicMetal
import LogicModel
import XCTest

/// Duplicated from `Tests/LogicModelTests/TestSupport.swift`: test targets are separate modules,
/// so `internal` helpers there are not visible here.
enum FixturePaths {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")
    static func parity(_ name: String) -> URL { root.appendingPathComponent("parity").appendingPathComponent(name) }
}

struct ParityCase {
    let model: LogicModelData
    let features: FeatureBatch
    let layers: [UInt8]        // [L,B,S,S,C]
    let expected: [String: Any]
    let boardSize: Int
    let batch: Int

    static func load(_ name: String) throws -> ParityCase {
        let dir = FixturePaths.parity(name)
        let inputs = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("inputs.json"))) as! [String: Any]
        let S = inputs["boardSize"] as! Int
        let B = inputs["batch"] as! Int
        let model = try ModelLoader.load(directory: dir.appendingPathComponent("model.ichigo"))
        let spatial = [UInt8](try Data(contentsOf: dir.appendingPathComponent("spatial.u8")))
        let globalData = try Data(contentsOf: dir.appendingPathComponent("global.f32"))
        let global: [Float] = globalData.withUnsafeBytes { buf in
            (0 ..< buf.count / 4).map { Float(bitPattern: UInt32(littleEndian: buf.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        let legal = [UInt8](try Data(contentsOf: dir.appendingPathComponent("legal.u8")))
        let layers = [UInt8](try Data(contentsOf: dir.appendingPathComponent("layers.u8")))
        let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("expected.json"))) as! [String: Any]
        let features = try FeatureBatch(boardSize: S, batch: B, spatial: spatial, global: global, legal: legal)
        return ParityCase(model: model, features: features, layers: layers, expected: expected, boardSize: S, batch: B)
    }

    func floats(_ key: String) -> [Float] {
        let v = expected[key]!
        if let rows = v as? [[Double]] { return rows.flatMap { $0.map(Float.init) } }
        return (v as! [Double]).map(Float.init)
    }
}

/// docs/spec/05-validation.md §3: |a-b| <= 1e-4 + 1e-4*|reference|
func assertHeadClose(_ got: [Float], _ ref: [Float], _ label: String, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(got.count, ref.count, "\(label) count", file: file, line: line)
    for i in 0 ..< min(got.count, ref.count) {
        let tol = 1e-4 + 1e-4 * abs(ref[i])
        let diff = abs(got[i] - ref[i])
        if diff > tol {
            XCTFail("\(label)[\(i)] got \(got[i]) expected \(ref[i]) (diff \(diff) > tol \(tol))", file: file, line: line)
            return
        }
    }
}

/// Skip cleanly (with a reason) rather than crash when this host has no Metal device
/// (docs/spec/05-validation.md §1: "必要GPUがない場合は明示的skip理由...を出し").
func requireMetal(file: StaticString = #filePath, line: UInt = #line) throws {
    guard MetalAvailability.probe().available else {
        throw XCTSkip("no Metal device available on this host", file: file, line: line)
    }
}

/// Deterministic, dependency-free PRNG for synthesising `FeatureBatch` inputs at arbitrary batch
/// sizes (SplitMix64).
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A structurally valid `FeatureBatch` with pseudo-random 0/1 spatial/legal bits and bounded
/// global floats. `layerOutputs`/`evaluate` never inspect `legal` for validity (only
/// `Postprocess` does), so this is safe to feed straight to a `LogicBackend`.
func syntheticFeatures(boardSize: Int, batch: Int, seed: UInt64) throws -> FeatureBatch {
    var rng = SplitMix64(seed: seed)
    let S = boardSize
    var spatial = [UInt8](repeating: 0, count: batch * S * S * 32)
    for i in 0 ..< spatial.count { spatial[i] = UInt8(rng.next() & 1) }
    var global = [Float](repeating: 0, count: batch * 4)
    for i in 0 ..< global.count { global[i] = Float(Int64(rng.next() % 2001) - 1000) / 1000 }
    var legal = [UInt8](repeating: 0, count: batch * (S * S + 1))
    for i in 0 ..< legal.count { legal[i] = UInt8(rng.next() & 1) }
    return try FeatureBatch(boardSize: S, batch: batch, spatial: spatial, global: global, legal: legal)
}
