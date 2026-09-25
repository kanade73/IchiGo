import IchiGoCore
import IchiGoFeatures
import XCTest

/// Pins the observable `Board` state, and the featureVersion 2 group planes computed from it, over
/// deterministic random games. Internal rewrites of `Board` (2026-09-24: play and chain rebuild
/// without `Set` allocations) must keep this digest bit-identical: chain heads and chain order feed
/// the ladder search, so a changed order could silently change features trained models rely on.
final class BoardGoldenTests: XCTestCase {
    private struct FNV {
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        mutating func add(_ v: UInt64) {
            var x = v
            for _ in 0 ..< 8 {
                h ^= x & 0xFF
                h = h &* 0x0000_0100_0000_01B3
                x >>= 8
            }
        }

        mutating func add(_ v: Int) { add(UInt64(bitPattern: Int64(v))) }
    }

    private func digest(size S: Int, games: Int) -> UInt64 {
        var f = FNV()
        var seed: UInt64 = 7 &+ UInt64(S)
        func next(_ n: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((seed >> 33) % UInt64(n))
        }
        for _ in 0 ..< games {
            let board = Board(S, S)
            var pla = Player.black
            for move in 0 ..< S * S * 2 {
                let legal = board.playableLocations().filter { board.colors[$0] == .empty && board.isLegal($0, pla) }
                board.playMove(legal.isEmpty || next(40) == 0 ? Board.passLoc : legal[next(legal.count)], pla)
                for loc in 0 ..< Board.maxArrSize {
                    let d = board.chainData[loc]
                    f.add(Int(board.colors[loc].rawValue))
                    f.add(board.chainHead[loc])
                    f.add(board.nextInChain[loc])
                    f.add(Int(d.owner.rawValue))
                    f.add(d.numLocs)
                    f.add(d.numLiberties)
                }
                f.add(board.koLoc)
                f.add(board.posHash.hash0)
                f.add(board.posHash.hash1)
                f.add(board.numBlackCaptures)
                f.add(board.numWhiteCaptures)
                pla = pla.opponent
                if move % 7 == 3 {
                    for bits in GroupFeatures.compute(board: board, toMove: pla) { f.add(Int(bits)) }
                }
            }
        }
        return f.h
    }

    func testBoardStateDigestIsUnchanged9() {
        XCTAssertEqual(digest(size: 9, games: 12), 1_244_839_556_423_362_139)
    }

    func testBoardStateDigestIsUnchanged19() {
        XCTAssertEqual(digest(size: 19, games: 3), 14_210_464_132_215_856_213)
    }
}
