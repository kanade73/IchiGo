import Foundation
import LogicModel
import XCTest

/// The Swift gate evaluation is `(g >> (2a+b)) & 1`; check all 64 truth values and the named ids.
final class GateTests: XCTestCase {
    func testAll64TruthValues() {
        let named: [(UInt8, [UInt8])] = [
            (0, [0, 0, 0, 0]), (8, [0, 0, 0, 1]), (6, [0, 1, 1, 0]), (14, [0, 1, 1, 1]),
            (12, [0, 0, 1, 1]), (10, [0, 1, 0, 1]), (3, [1, 1, 0, 0]), (7, [1, 1, 1, 0]), (15, [1, 1, 1, 1]),
        ]
        for (g, rows) in named {
            for (i, expected) in rows.enumerated() {
                XCTAssertEqual((g >> UInt8(i)) & 1, expected, "gate \(g) row \(i)")
            }
        }
        for g in 0 ..< 16 {
            for a in 0 ..< 2 {
                for b in 0 ..< 2 {
                    XCTAssertEqual((UInt8(g) >> UInt8(2 * a + b)) & 1, UInt8((g >> (2 * a + b)) & 1))
                }
            }
        }
    }

    func testFeatureBatchValidation() {
        XCTAssertThrowsError(try FeatureBatch(boardSize: 9, batch: 1, spatial: [UInt8](repeating: 2, count: 81 * 32), global: [0, 0, 0, 0], legal: [UInt8](repeating: 1, count: 82)))
        XCTAssertThrowsError(try FeatureBatch(boardSize: 9, batch: 1, spatial: [UInt8](repeating: 0, count: 81 * 32 - 1), global: [0, 0, 0, 0], legal: [UInt8](repeating: 1, count: 82)))
        XCTAssertThrowsError(try FeatureBatch(boardSize: 9, batch: 1, spatial: [UInt8](repeating: 0, count: 81 * 32), global: [0, .nan, 0, 0], legal: [UInt8](repeating: 1, count: 82)))
    }
}
