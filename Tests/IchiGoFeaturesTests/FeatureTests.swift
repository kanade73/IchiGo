import IchiGoCore
import IchiGoFeatures
import XCTest

/// Hand-computed fixtures for docs/spec/01-network.md §1 (T04 acceptance).
final class FeatureTests: XCTestCase {
    /// Expected spatial planes as `[channel: set of point indices that are 1]`; every other
    /// (point, channel) must be 0. `full` lists channels that are 1 everywhere.
    private func assertPlanes(
        _ enc: FeatureEncoder.Encoded, size S: Int, ones: [Int: Set<Int>], full: Set<Int>,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(enc.batch, 1, file: file, line: line)
        let ring: Set<Int> = Set((0 ..< S * S).filter { $0 % S == 0 || $0 % S == S - 1 || $0 / S == 0 || $0 / S == S - 1 })
        for c in 0 ..< 32 {
            for p in 0 ..< (S * S) {
                let v = enc.spatial[p * 32 + c]
                var expected: UInt8 = 0
                if full.contains(c) || c == 28 { expected = 1 }
                if c == 27, ring.contains(p) { expected = 1 }
                if let s = ones[c], s.contains(p) { expected = 1 }
                if v != expected {
                    XCTFail("channel \(c) point \(p) (x=\(p % S),y=\(p / S)) got \(v) expected \(expected)", file: file, line: line)
                    return
                }
                XCTAssertTrue(v == 0 || v == 1, file: file, line: line)
            }
        }
    }

    private func idx(_ x: Int, _ y: Int, _ S: Int) -> Int { y * S + x }

    private func assertGlobal(_ got: [Float], _ expected: [Double], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for (g, e) in zip(got, expected) { XCTAssertEqual(g, Float(e), accuracy: 1e-7, file: file, line: line) }
    }

    func testEmptyBoard9() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        let enc = try FeatureEncoder.encode([g.snapshot()])
        assertPlanes(enc, size: 9, ones: [:], full: [16, 17, 31])
        assertGlobal(enc.global, [-7.0 / 81.0, 9.0 / 19.0, 0, 0])
        XCTAssertEqual(enc.legal, [UInt8](repeating: 1, count: 82))
    }

    func testEmptyBoard19WhiteToMove() throws {
        let g = try GameState(boardSize: 19, komi: 7.5, initialPlayer: .white)
        let enc = try FeatureEncoder.encode([g.snapshot()])
        assertPlanes(enc, size: 19, ones: [:], full: [16, 17])
        assertGlobal(enc.global, [7.5 / 361.0, 1, 0, 0])
    }

    func testCapture9() throws {
        let S = 9
        let g = try GameState(boardSize: S, komi: 7)
        try g.play(.black, .point(x: 4, y: 4))
        try g.play(.white, .point(x: 4, y: 5))
        try g.play(.black, .point(x: 3, y: 5))
        try g.play(.white, .pass)
        try g.play(.black, .point(x: 5, y: 5))
        try g.play(.white, .pass)
        try g.play(.black, .point(x: 4, y: 6))  // captures white (4,5)
        let snap = g.snapshot()
        XCTAssertEqual(snap.toMove, .white)
        XCTAssertEqual(snap.moveNumber, 7)
        XCTAssertNil(snap.koPoint)
        let enc = try FeatureEncoder.encode([snap])
        let b4: Set<Int> = [idx(4, 4, S), idx(3, 5, S), idx(5, 5, S), idx(4, 6, S)]
        let b3: Set<Int> = [idx(4, 4, S), idx(3, 5, S), idx(5, 5, S)]
        let b2: Set<Int> = [idx(4, 4, S), idx(3, 5, S)]
        let w1: Set<Int> = [idx(4, 5, S)]
        let all = Set(0 ..< S * S)
        assertPlanes(enc, size: S, ones: [
            1: b4,                       // opponent (black) now
            2: w1, 3: b3,                // t=1: before B(4,6)
            4: w1, 5: b3,                // t=2: after B(5,5) (then W pass)
            6: w1, 7: b2,                // t=3: after W pass #1
            8: w1, 9: b2,                // t=4: after B(3,5)
            10: w1, 11: [idx(4, 4, S)],  // t=5: after W(4,5)
            13: [idx(4, 4, S)],          // t=6: after B(4,4)
            16: all.subtracting(b4),
            17: all.subtracting(b4).subtracting([idx(4, 5, S)]),  // (4,5) is suicide for white
            24: b4,                      // black chains all have >=3 liberties
            25: [idx(4, 6, S)],
        ], full: [30])
        assertGlobal(enc.global, [7.0 / 81.0, 9.0 / 19.0, 7.0 / 162.0, 0])
        XCTAssertEqual(enc.legal[idx(4, 5, S)], 0)
        XCTAssertEqual(enc.legal[81], 1)
    }

    func testSimpleKo9() throws {
        let S = 9
        let g = try GameState(boardSize: S, komi: 7, initialStones: [
            (.black, 1, 1), (.black, 2, 0), (.black, 2, 2), (.white, 3, 0), (.white, 4, 1), (.white, 3, 2),
        ])
        try g.play(.black, .point(x: 3, y: 1))
        try g.play(.white, .point(x: 2, y: 1))  // captures (3,1): ko
        let snap = g.snapshot()
        XCTAssertEqual(snap.toMove, .black)
        XCTAssertEqual(snap.koPoint, idx(3, 1, S))
        XCTAssertFalse(g.isLegal(.black, .point(x: 3, y: 1)))
        let enc = try FeatureEncoder.encode([snap])
        let blacks: Set<Int> = [idx(1, 1, S), idx(2, 0, S), idx(2, 2, S)]
        let whites3: Set<Int> = [idx(3, 0, S), idx(4, 1, S), idx(3, 2, S)]
        let whites4 = whites3.union([idx(2, 1, S)])
        let all = Set(0 ..< S * S)
        assertPlanes(enc, size: S, ones: [
            0: blacks, 1: whites4,
            2: blacks.union([idx(3, 1, S)]), 3: whites3,
            4: blacks, 5: whites3,
            16: all.subtracting(blacks).subtracting(whites4),
            17: all.subtracting(blacks).subtracting(whites4).subtracting([idx(3, 1, S)]),
            18: [idx(3, 1, S)],
            19: [idx(2, 0, S)], 20: [idx(2, 2, S)], 21: [idx(1, 1, S)],
            22: [idx(2, 1, S)], 23: [idx(3, 0, S)], 24: [idx(4, 1, S), idx(3, 2, S)],
            25: [idx(2, 1, S)], 26: [idx(3, 1, S)],
        ], full: [31])
        assertGlobal(enc.global, [-7.0 / 81.0, 9.0 / 19.0, 2.0 / 162.0, 0])
    }

    func testSuicide19() throws {
        let S = 19
        let g = try GameState(boardSize: S, komi: 7.5, initialStones: [(.black, 1, 0), (.black, 0, 1)], initialPlayer: .white)
        let enc = try FeatureEncoder.encode([g.snapshot()])
        let blacks: Set<Int> = [idx(1, 0, S), idx(0, 1, S)]
        let all = Set(0 ..< S * S)
        assertPlanes(enc, size: S, ones: [
            1: blacks,
            16: all.subtracting(blacks),
            17: all.subtracting(blacks).subtracting([idx(0, 0, S)]),
            24: blacks,
        ], full: [])
        XCTAssertEqual(enc.legal[idx(0, 0, S)], 0)
        XCTAssertEqual(enc.legal[361], 1)
    }

    func testTwoPasses9() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .pass)
        try g.play(.white, .pass)
        let snap = g.snapshot()
        XCTAssertTrue(snap.isGameFinished)
        XCTAssertEqual(snap.consecutivePasses, 2)
        let enc = try FeatureEncoder.encode([snap])
        assertPlanes(enc, size: 9, ones: [:], full: [16, 17, 29, 30, 31])
        assertGlobal(enc.global, [-7.0 / 81.0, 9.0 / 19.0, 2.0 / 162.0, 1])
    }

    func testHistoryShorterThanSevenIsZero() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .point(x: 4, y: 4))
        let enc = try FeatureEncoder.encode([g.snapshot()])
        // t=1 layout is empty; t>=2 absent → channels 4..15 all zero
        for c in 4 ..< 16 { for p in 0 ..< 81 { XCTAssertEqual(enc.spatial[p * 32 + c], 0) } }
        XCTAssertEqual(enc.spatial[(4 * 9 + 4) * 32 + 1], 1)  // opponent's stone (white to move)
    }

    func testSnapshotIsIndependentOfLaterMoves() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .point(x: 2, y: 2))
        let before = g.snapshot()
        try g.play(.white, .point(x: 6, y: 6))
        try g.play(.black, .point(x: 2, y: 6))
        let again = before
        XCTAssertEqual(before, again)
        XCTAssertEqual(before.current[2 * 9 + 2], 1)
        XCTAssertEqual(before.current[6 * 9 + 6], 0)
        XCTAssertEqual(before.moveNumber, 1)
        try g.undo()
        try g.undo()
        XCTAssertEqual(g.snapshot(), before)
    }

    /// RinGo's "sending two, returning one" reference position (RulesReferenceTests
    /// testPositionalKoRules) embedded at the right edge of a 9x9 board with a white wall in
    /// column 2 so the pattern keeps its original liberties. After the sequence, both (3,0) and
    /// (8,1) are illegal for black by positional superko only (no simple-ko point), so channel 17
    /// must be 0 there while channel 18 stays all zero.
    func testPositionalSuperkoBanIsInLegalPlane() throws {
        let S = 9
        let rows = [".o.xxo", "oxxxo.", "o.x.oo", "xx.oo.", "oooo.o"]
        var stones: [(player: Player, x: Int, y: Int)] = []
        for (y, r) in rows.enumerated() {
            for (x, ch) in r.enumerated() {
                if ch == "x" { stones.append((.black, x + 3, y)) } else if ch == "o" { stones.append((.white, x + 3, y)) }
            }
        }
        for y in 0 ..< 5 { stones.append((.white, 2, y)) }
        let g = try GameState(boardSize: S, komi: 7, initialStones: stones)
        let seq: [(Player, MoveCoord)] = [
            (.black, .point(x: 8, y: 1)), (.white, .pass), (.black, .point(x: 6, y: 2)), (.white, .point(x: 5, y: 0)),
            (.black, .point(x: 3, y: 0)), (.white, .point(x: 8, y: 0)), (.black, .pass), (.white, .point(x: 4, y: 0)),
            (.black, .pass), (.white, .point(x: 5, y: 0)),
        ]
        for (p, m) in seq { try g.play(p, m) }
        let snap = g.snapshot()
        XCTAssertEqual(snap.toMove, .black)
        XCTAssertNil(snap.koPoint)
        XCTAssertFalse(snap.isGameFinished)
        XCTAssertEqual(snap.legal[idx(3, 0, S)], 0)
        XCTAssertEqual(snap.legal[idx(8, 1, S)], 0)
        XCTAssertEqual(snap.current[idx(3, 0, S)], 0)
        XCTAssertEqual(snap.current[idx(8, 1, S)], 0)
        let enc = try FeatureEncoder.encode([snap])
        XCTAssertEqual(enc.spatial[idx(3, 0, S) * 32 + 17], 0)
        XCTAssertEqual(enc.spatial[idx(8, 1, S) * 32 + 17], 0)
        XCTAssertEqual(enc.spatial[idx(3, 0, S) * 32 + 16], 1)
        for p in 0 ..< (S * S) { XCTAssertEqual(enc.spatial[p * 32 + 18], 0) }
        // legality plane equals the rules oracle everywhere
        for y in 0 ..< S {
            for x in 0 ..< S {
                XCTAssertEqual(enc.spatial[idx(x, y, S) * 32 + 17], g.isLegal(.black, .point(x: x, y: y)) ? 1 : 0)
            }
        }
    }

    // MARK: - Fingerprint / coordinate validation regressions

    /// Same 4 stones placed in a different order: the boards match but the move history (and hence
    /// the encoded history planes) differ, so the fingerprints must differ too.
    func testFingerprintCoversMoveOrderAndHistory() throws {
        let pts: [MoveCoord] = [.point(x: 2, y: 2), .point(x: 6, y: 6), .point(x: 2, y: 6), .point(x: 6, y: 2)]
        func playOrder(_ order: [Int]) throws -> GameState {
            let g = try GameState(boardSize: 9, komi: 7)
            for (i, k) in order.enumerated() { try g.play(i % 2 == 0 ? .black : .white, pts[k]) }
            return g
        }
        let a = try playOrder([0, 1, 2, 3]).snapshot()
        let b = try playOrder([2, 1, 0, 3]).snapshot()
        XCTAssertEqual(a.current, b.current)          // identical board
        XCTAssertNotEqual(a.fingerprint, b.fingerprint)
        // ... and the encoded features really do differ, so the fingerprint must not collide.
        let ea = try FeatureEncoder.encode([a]), eb = try FeatureEncoder.encode([b])
        XCTAssertNotEqual(ea.spatial, eb.spatial)
        // Identical sequences produce identical fingerprints.
        let c = try playOrder([0, 1, 2, 3]).snapshot()
        XCTAssertEqual(a.fingerprint, c.fingerprint)
        XCTAssertEqual(try FeatureEncoder.encode([c]).spatial, ea.spatial)
        // A transposition further back than the 2 recent moves is still distinguished.
        let d = try playOrder([1, 0, 2, 3]).snapshot()
        XCTAssertEqual(a.recentMoves, d.recentMoves)
        XCTAssertNotEqual(a.fingerprint, d.fingerprint)
        XCTAssertEqual(a.fingerprint.count, 64)
    }

    func testOffBoardCoordinatesRejected() throws {
        let g = try GameState(boardSize: 9, komi: 7)
        for m: MoveCoord in [.point(x: 11, y: 0), .point(x: -1, y: 0), .point(x: 0, y: 9), .point(x: 0, y: -3)] {
            XCTAssertFalse(g.isLegal(.black, m), "\(m) must be illegal")
            XCTAssertThrowsError(try g.play(.black, m)) { e in
                guard case let GameStateError.illegalMove(_, _, reason) = e else { return XCTFail("\(e)") }
                XCTAssertEqual(reason, "off-board")
            }
        }
        XCTAssertEqual(g.moveNumber, 0)
        XCTAssertTrue(g.isLegal(.black, .pass))
    }

    func testDuplicateInitialStonesRejected() throws {
        for stones: [(player: Player, x: Int, y: Int)] in [
            [(.black, 2, 2), (.white, 2, 2)],
            [(.black, 2, 2), (.black, 2, 2)],
        ] {
            XCTAssertThrowsError(try GameState(boardSize: 9, komi: 7, initialStones: stones)) { e in
                XCTAssertEqual(e as? GameStateError, GameStateError.invalidInitialStone(.point(x: 2, y: 2)))
            }
        }
        XCTAssertNoThrow(try GameState(boardSize: 9, komi: 7, initialStones: [(.black, 2, 2), (.white, 3, 2)]))
    }

    func testMixedSizesRejected() throws {
        let a = try GameState(boardSize: 9, komi: 7).snapshot()
        let b = try GameState(boardSize: 19, komi: 7.5).snapshot()
        XCTAssertThrowsError(try FeatureEncoder.encode([a, b]))
    }
}
