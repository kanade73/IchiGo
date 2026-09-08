import Foundation
import IchiGoFeatures
import XCTest

final class SymmetryTests: XCTestCase {
    private func fixture(_ size: Int) throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/symmetry/perm-\(size).json")
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }

    func testMatchesSharedFixture() throws {
        for size in [9, 19] {
            let f = try fixture(size)
            let sym = FeatureSymmetry(boardSize: size)
            XCTAssertEqual(sym.forward, f["forward"] as! [[Int]], "forward \(size)")
            XCTAssertEqual(sym.inverse, f["inverse"] as! [[Int]], "inverse \(size)")
        }
    }

    func testDefinitions() {
        XCTAssertEqual(FeatureSymmetry.map(x: 0, y: 0, size: 9, sym: 1).0, 8)
        XCTAssertEqual(FeatureSymmetry.map(x: 0, y: 0, size: 9, sym: 1).1, 0)
        XCTAssertEqual(FeatureSymmetry.map(x: 2, y: 5, size: 9, sym: 4).0, 6)
        XCTAssertEqual(FeatureSymmetry.map(x: 2, y: 5, size: 9, sym: 4).1, 5)
    }

    func testAllEightRoundTripAndPassInvariant() {
        for size in [9, 19] {
            let sym = FeatureSymmetry(boardSize: size)
            let P = size * size
            var rng = SystemRandomNumberGenerator()
            let spatial = (0 ..< 2 * P * 32).map { _ in UInt8.random(in: 0 ... 1, using: &rng) }
            let policy = (0 ..< P + 1).map { Float($0) }
            let plane = (0 ..< P).map { Float($0) * 0.5 }
            for s in 0 ..< 8 {
                let t = sym.transformSpatial(spatial, batch: 2, channels: 32, sym: s)
                XCTAssertEqual(sym.transformSpatial(t, batch: 2, channels: 32, sym: s, inverse: true), spatial)
                let tp = sym.transformPolicy(policy, sym: s)
                XCTAssertEqual(tp[P], policy[P])
                XCTAssertEqual(sym.transformPolicy(tp, sym: s, inverse: true), policy)
                XCTAssertEqual(sym.transformPlane(sym.transformPlane(plane, sym: s), sym: s, inverse: true), plane)
                XCTAssertEqual(Set(sym.forward[s]).count, P)
            }
            XCTAssertEqual(sym.forward[0], Array(0 ..< P))
        }
    }
}
