import IchiGoCore
import IchiGoFeatures
import XCTest

final class SGFReplayTests: XCTestCase {
    func testCanonicalHashMatchesPython() {
        // Values computed with Python: json.dumps(obj, sort_keys=True, separators=(",",":")) + sha256
        XCTAssertEqual(
            PositionExport.canonicalHash(boardSize: 9, komi: 7, initialStones: [], initialPlayer: "B", moves: [["B", "E5"], ["W", "pass"]]),
            "9d7a9940b43c10dd93748dd347fa87d79b148940d8c8806a79a34763a3158310"
        )
        XCTAssertEqual(
            PositionExport.canonicalHash(boardSize: 9, komi: 7.5, initialStones: [["B", "aa"], ["W", "cc"]], initialPlayer: "B", moves: [["B", "E5"], ["W", "pass"]]),
            "8c0332ec581a41ee493d94c45cdf0476becd2e3af044c1138cc37433f8c3b687"
        )
    }

    func testReplayCaptureAndPassMatchesSnapshotFixture() throws {
        // Same game as FeatureTests.testCapture9, expressed as SGF.
        let sgf = "(;GM[1]SZ[9]KM[7]RU[Chinese];B[ee];W[ef];B[df];W[];B[ff];W[];B[eg])"
        let r = try PositionExport.replay(sgf: sgf, expectedSize: 9)
        XCTAssertEqual(r.moves.count, 7)
        XCTAssertEqual(r.moves[0], ["B", "E5"])
        XCTAssertEqual(r.moves[3], ["W", "pass"])
        let last = r.snapshots[7]
        XCTAssertEqual(last.toMove, .white)
        XCTAssertEqual(last.current[5 * 9 + 4], 0)  // captured
        XCTAssertEqual(last.legal[5 * 9 + 4], 0)    // suicide for white
        XCTAssertEqual(last.recentMoves, [.point(x: 4, y: 6), .pass])
        // Same as the hand-built GameState
        let g = try GameState(boardSize: 9, komi: 7)
        for m in [(Player.black, MoveCoord.point(x: 4, y: 4)), (.white, .point(x: 4, y: 5)), (.black, .point(x: 3, y: 5)), (.white, .pass), (.black, .point(x: 5, y: 5)), (.white, .pass), (.black, .point(x: 4, y: 6))] {
            try g.play(m.0, m.1)
        }
        XCTAssertEqual(g.snapshot(), last)
        let row = try PositionExport.row(r, turn: 7)
        XCTAssertEqual(row["turnNumber"] as? Int, 7)
        XCTAssertEqual(row["toMove"] as? String, "W")
        XCTAssertEqual((row["moves"] as? [[String]])?.count, 7)
        XCTAssertEqual((row["spatial"] as? [Int])?.count, 81 * 32)
        XCTAssertEqual(row["positionId"] as? String, PositionExport.canonicalHash(boardSize: 9, komi: 7, initialStones: [], initialPlayer: "B", moves: r.moves))
        XCTAssertEqual(row["gameId"] as? String, r.gameId)
        XCTAssertNotEqual(try PositionExport.row(r, turn: 3)["positionId"] as? String, row["positionId"] as? String)
    }

    func testRejectsSizeMismatchIllegalMoveAndVariations() {
        XCTAssertThrowsError(try PositionExport.replay(sgf: "(;SZ[19];B[aa])", expectedSize: 9)) { e in
            guard case PositionExport.RejectReason.boardSize = e else { return XCTFail("\(e)") }
        }
        // occupied point
        XCTAssertThrowsError(try PositionExport.replay(sgf: "(;SZ[9];B[aa];W[aa])", expectedSize: 9)) { e in
            guard case let PositionExport.RejectReason.illegalMove(index, p, m, _) = e else { return XCTFail("\(e)") }
            XCTAssertEqual(index, 1); XCTAssertEqual(p, "W"); XCTAssertEqual(m, "A9")
        }
        // suicide: white into a corner surrounded by black
        XCTAssertThrowsError(try PositionExport.replay(sgf: "(;SZ[9]AB[ba][ab];W[aa])", expectedSize: 9)) { e in
            guard case PositionExport.RejectReason.illegalMove = e else { return XCTFail("\(e)") }
        }
        // simple ko retake is illegal (also superko)
        let ko = "(;SZ[9]AB[bb][ca][cc]AW[da][eb][dc];B[db];W[cb];B[db])"
        XCTAssertThrowsError(try PositionExport.replay(sgf: ko, expectedSize: 9)) { e in
            guard case let PositionExport.RejectReason.illegalMove(index, _, _, _) = e else { return XCTFail("\(e)") }
            XCTAssertEqual(index, 2)
        }
        XCTAssertThrowsError(try PositionExport.replay(sgf: "(;SZ[9];B[aa](;W[bb])(;W[cc]))", expectedSize: 9)) { e in
            guard case PositionExport.RejectReason.parse = e else { return XCTFail("\(e)") }
        }
    }

    /// The same point set up as both black and white must be rejected, not silently overwritten.
    func testDuplicateSetupStoneRejected() {
        XCTAssertThrowsError(try PositionExport.replay(sgf: "(;SZ[9]AB[aa]AW[aa])", expectedSize: 9)) { e in
            guard case let PositionExport.RejectReason.invalidInitialStone(m) = e else { return XCTFail("\(e)") }
            XCTAssertTrue(m.contains("invalid initial stone"), m)
        }
    }

    func testInitialStonesSortedAndHandicapPlayer() throws {
        let r = try PositionExport.replay(sgf: "(;SZ[9]KM[0]AB[cc][gg][ec]HA[3])", expectedSize: 9)
        XCTAssertEqual(r.initialStones, [["B", "cc"], ["B", "ec"], ["B", "gg"]])
        XCTAssertEqual(r.initialPlayer, "W")
        XCTAssertEqual(r.snapshots[0].toMove, .white)
        XCTAssertEqual(r.snapshots[0].moveNumber, 0)
    }
}
