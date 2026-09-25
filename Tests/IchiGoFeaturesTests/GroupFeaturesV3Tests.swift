import IchiGoCore
import IchiGoFeatures
import XCTest

/// Hand-built positions for the featureVersion 3 planes (`GroupFeaturesV3`). Diagrams are rows top
/// (y=0) to bottom, `X` black, `O` white.
final class GroupFeaturesV3Tests: XCTestCase {
    private func layout(_ rows: [String]) -> (StoneLayout, Int) {
        let S = rows.count
        var out = StoneLayout(repeating: 0, count: S * S)
        for (y, row) in rows.enumerated() {
            let cells = row.split(separator: " ")
            XCTAssertEqual(cells.count, S)
            for (x, c) in cells.enumerated() { out[y * S + x] = c == "X" ? 1 : c == "O" ? 2 : 0 }
        }
        return (out, S)
    }

    private func points(_ bits: [UInt8], _ S: Int, bit: Int) -> Set<String> {
        Set(bits.indices.filter { bits[$0] & (UInt8(1) << UInt8(bit)) != 0 }.map { "\($0 % S),\($0 / S)" })
    }

    private let empty9 = ". . . . . . . . ."

    /// Two single-point eyes: eye credit 2; the outside region touches both colours, so it gives
    /// none. Filling one eye leaves credit 1.
    func testTwoEyesGiveCreditTwoAndOneEyeDoesNot() {
        let rows = [". X . X . . . . .", "X X X X . . . . ."] + Array(repeating: empty9, count: 6) + [". . . . . . . . O"]
        let (l, S) = layout(rows)
        let group: Set<String> = ["1,0", "3,0", "0,1", "1,1", "2,1", "3,1"]
        let bits = GroupFeaturesV3.compute(layout: l, size: S)
        XCTAssertEqual(points(bits, S, bit: 1), group)
        XCTAssertEqual(points(bits, S, bit: 0), group)   // 8 liberties; the lone O stone has 2
        var filled = l
        filled[2] = 1
        let one = GroupFeaturesV3.compute(layout: filled, size: S)
        XCTAssertEqual(points(one, S, bit: 1), [])
    }

    /// Enclosed regions of 3 points in a line have one vital point (credit 1); 4 points in a line
    /// split two ways (credit 2).
    func testStraightThreeIsOneEyeAndStraightFourIsTwo() {
        let three = [". . . X . . . . .", "X X X X . . . . ."] + Array(repeating: empty9, count: 6) + [". . . . . . . . O"]
        let (l3, S) = layout(three)
        XCTAssertEqual(points(GroupFeaturesV3.compute(layout: l3, size: S), S, bit: 1), [])
        let four = [". . . . X . . . .", "X X X X X . . . ."] + Array(repeating: empty9, count: 6) + [". . . . . . . . O"]
        let (l4, _) = layout(four)
        XCTAssertEqual(points(GroupFeaturesV3.compute(layout: l4, size: S), S, bit: 1), ["4,0", "0,1", "1,1", "2,1", "3,1", "4,1"])
    }

    /// A 2-liberty stone next to a 3-liberty enemy chain is marked; the enemy chain is not.
    func testWeakerChainTouchingStrongerEnemyIsMarked() {
        let rows = [". . . . . . . . .", ". . . . . . . . .", ". . . O O . . . .", ". . . X . . . . .", ". . O . . . . . ."]
            + Array(repeating: empty9, count: 4)
        let (l, S) = layout(rows)
        let bits = GroupFeaturesV3.compute(layout: l, size: S)
        // X at (3,3): 3 liberties, next to the O chain (3,2),(4,2) with 5; O at (2,4) touches no X.
        XCTAssertEqual(points(bits, S, bit: 2), ["3,3"])
    }

    /// A stone in atari is capturable; a 2-liberty stone that can extend to 3 liberties is not.
    func testCaptureSearch() {
        let atari = ["X O . . . . . . ."] + Array(repeating: empty9, count: 8)
        let (la, S) = layout(atari)
        XCTAssertEqual(points(GroupFeaturesV3.compute(layout: la, size: S), S, bit: 3), ["0,0"])
        let open = Array(repeating: empty9, count: 4) + [". . . O X O . . ."] + Array(repeating: empty9, count: 4)
        let (lo, _) = layout(open)
        XCTAssertFalse(points(GroupFeaturesV3.compute(layout: lo, size: S), S, bit: 3).contains("4,4"))
    }

    /// v3 keeps every channel of v2 except history t=2 (4, 5), empty (16) and edge ring (27).
    func testEncoderV3ReplacesFourChannels() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        for (x, y) in [(4, 4), (2, 2), (6, 6), (2, 6), (6, 2), (3, 3)] { try g.play(g.toMove, .point(x: x, y: y)) }
        let snap = g.snapshot()
        let v2 = try FeatureEncoder.encode([snap], featureVersion: 2)
        let v3 = try FeatureEncoder.encode([snap], featureVersion: 3)
        let planes = GroupFeaturesV3.compute(layout: snap.current, size: 9)
        for p in 0 ..< 81 {
            for c in 0 ..< 32 {
                if let i = GroupFeaturesV3.channels.firstIndex(of: c) {
                    XCTAssertEqual(v3.spatial[p * 32 + c], planes[p] & (UInt8(1) << UInt8(i)) != 0 ? 1 : 0, "channel \(c) point \(p)")
                } else {
                    XCTAssertEqual(v3.spatial[p * 32 + c], v2.spatial[p * 32 + c], "channel \(c) point \(p)")
                }
            }
        }
        XCTAssertEqual(v2.global, v3.global)
        XCTAssertEqual(v2.legal, v3.legal)
        let batch = try FeatureEncoder.encode(Array(repeating: snap, count: 12), featureVersion: 3)
        XCTAssertEqual(Array(batch.spatial.prefix(81 * 32)), v3.spatial)   // parallel batch path
    }
}
