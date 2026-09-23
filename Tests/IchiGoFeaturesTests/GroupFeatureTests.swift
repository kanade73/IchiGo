import IchiGoCore
import IchiGoFeatures
import XCTest

/// Hand-built positions for the featureVersion 2 group planes (docs/spec/01-network.md §1,
/// `GroupFeatures`). Diagrams are rows top (y=0) to bottom, `X` black, `O` white.
final class GroupFeatureTests: XCTestCase {
    private func layout(_ rows: [String]) -> (StoneLayout, Int) {
        let S = rows.count
        var out = StoneLayout(repeating: 0, count: S * S)
        for (y, row) in rows.enumerated() {
            let cells = row.split(separator: " ")
            XCTAssertEqual(cells.count, S)
            for (x, c) in cells.enumerated() {
                out[y * S + x] = c == "X" ? 1 : c == "O" ? 2 : 0
            }
        }
        return (out, S)
    }

    /// Points (as "x,y") whose bit `bit` is set.
    private func points(_ bits: [UInt16], _ S: Int, bit: Int) -> Set<String> {
        Set(bits.indices.filter { bits[$0] & (UInt16(1) << UInt16(bit)) != 0 }.map { "\($0 % S),\($0 / S)" })
    }

    private let empty9 = ". . . . . . . . ."

    func testOneLibertyCornerStoneIsLadderCaptured() {
        let (l, S) = layout([
            "O X . . . . . . .",
            ". X . . . . . . .",
        ] + Array(repeating: empty9, count: 7))
        let bits = GroupFeatures.compute(layout: l, size: S, toMove: .black)
        XCTAssertEqual(points(bits, S, bit: 1), ["0,0"])   // opponent (white) stone, captured even moving first
        XCTAssertEqual(points(bits, S, bit: 0), [])
        XCTAssertEqual(points(bits, S, bit: 2), [])         // working moves are only for 2-liberty chains
    }

    func testTwoLibertyCornerStoneHasWorkingLadderMoves() {
        let (l, S) = layout([
            "O . . . . . . . .",
            ". X . . . . . . .",
        ] + Array(repeating: empty9, count: 7))
        let bits = GroupFeatures.compute(layout: l, size: S, toMove: .black)
        XCTAssertEqual(points(bits, S, bit: 1), ["0,0"])
        XCTAssertEqual(points(bits, S, bit: 2), ["1,0", "0,1"])
        // same board, white to move: the white stone is now "own", and working moves are black's only
        let w = GroupFeatures.compute(layout: l, size: S, toMove: .white)
        XCTAssertEqual(points(w, S, bit: 0), ["0,0"])
        XCTAssertEqual(points(w, S, bit: 1), [])
        XCTAssertEqual(points(w, S, bit: 2), [])
    }

    func testTwoEyedCornerGroupIsPassAliveWithEyesAndTwoEyeMark() {
        let (l, S) = layout([
            ". X . X . . . . .",
            "X X X X . . . . .",
        ] + Array(repeating: empty9, count: 7))
        let stones: Set<String> = ["1,0", "3,0", "0,1", "1,1", "2,1", "3,1"]
        let eyes: Set<String> = ["0,0", "2,0"]
        let bits = GroupFeatures.compute(layout: l, size: S, toMove: .black)
        XCTAssertEqual(points(bits, S, bit: 3), stones.union(eyes))  // own pass-alive area
        XCTAssertEqual(points(bits, S, bit: 4), [])
        XCTAssertEqual(points(bits, S, bit: 5), stones)              // 8 liberties
        XCTAssertEqual(points(bits, S, bit: 7), eyes)
        XCTAssertEqual(points(bits, S, bit: 9), stones)
        XCTAssertEqual(points(bits, S, bit: 0).union(points(bits, S, bit: 1)), [])

        let w = GroupFeatures.compute(layout: l, size: S, toMove: .white)
        XCTAssertEqual(points(w, S, bit: 4), stones.union(eyes))
        XCTAssertEqual(points(w, S, bit: 6), stones)
        XCTAssertEqual(points(w, S, bit: 8), eyes)
        XCTAssertEqual(points(w, S, bit: 9), stones)                 // colour-blind: marks the chain itself
    }

    func testFalseEyeIsNotAnEyeRegion() {
        let (l, S) = layout([
            empty9, empty9, empty9,
            ". . . O X . . . .",
            ". . . X . X . . .",
            ". . . . X O . . .",
            empty9, empty9, empty9,
        ])
        let bits = GroupFeatures.compute(layout: l, size: S, toMove: .black)
        XCTAssertFalse(points(bits, S, bit: 7).contains("4,4"))   // two white diagonals: false eye
        let (real, _) = layout([
            empty9, empty9, empty9,
            ". . . . X . . . .",
            ". . . X . X . . .",
            ". . . . X O . . .",
            empty9, empty9, empty9,
        ])
        XCTAssertTrue(points(GroupFeatures.compute(layout: real, size: S, toMove: .black), S, bit: 7).contains("4,4"))
    }

    func testEncoderV2KeepsTwoMovesOfHistoryAndWritesGroupPlanes() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        for (x, y) in [(4, 4), (2, 2), (6, 6), (2, 6), (6, 2)] { try g.play(g.toMove, .point(x: x, y: y)) }
        let snap = g.snapshot()
        let v1 = try FeatureEncoder.encode([snap])
        let v2 = try FeatureEncoder.encode([snap], featureVersion: 2)
        let S = 9
        for p in 0 ..< S * S {
            for c in 0 ..< 32 {
                let a = v1.spatial[p * 32 + c], b = v2.spatial[p * 32 + c]
                if c < 6 || c >= 16 { XCTAssertEqual(a, b, "channel \(c) point \(p) must not depend on the version") }
            }
        }
        XCTAssertEqual(v1.global, v2.global)
        XCTAssertEqual(v1.legal, v2.legal)
        // v1 has history 3+ moves back in channels 6...15; v2 has none there on this quiet board
        // except the 4+ liberty planes (every stone here has 4 liberties)
        let v1History = (0 ..< S * S).contains { p in (6 ..< 16).contains { v1.spatial[p * 32 + $0] == 1 } }
        XCTAssertTrue(v1History)
        let bits = GroupFeatures.compute(layout: snap.current, size: S, toMove: snap.toMove)
        for p in 0 ..< S * S {
            for i in 0 ..< GroupFeatures.planes {
                let expected: UInt8 = bits[p] & (UInt16(1) << UInt16(i)) != 0 ? 1 : 0
                XCTAssertEqual(v2.spatial[p * 32 + GroupFeatures.firstChannel + i], expected)
            }
        }
        XCTAssertThrowsError(try FeatureEncoder.encode([snap], featureVersion: 3))
    }
}
