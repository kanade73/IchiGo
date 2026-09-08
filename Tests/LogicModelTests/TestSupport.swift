import Foundation
import LogicModel
import XCTest

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
    var worst: Float = 0
    for i in 0 ..< min(got.count, ref.count) {
        let tol = 1e-4 + 1e-4 * abs(ref[i])
        let diff = abs(got[i] - ref[i])
        worst = max(worst, diff - tol)
        if diff > tol {
            XCTFail("\(label)[\(i)] got \(got[i]) expected \(ref[i]) (diff \(diff) > tol \(tol))", file: file, line: line)
            return
        }
    }
}
