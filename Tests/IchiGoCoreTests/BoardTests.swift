@testable import IchiGoCore
import XCTest

final class BoardTests: XCTestCase {
    func testStonePlacementCaptureAndChainMerge() {
        let board = Board(5, 5)
        let first = board.getLoc(1, 1)
        let second = board.getLoc(2, 1)
        XCTAssertTrue(board.playMove(first, .black, false))
        XCTAssertTrue(board.playMove(second, .black, false))
        XCTAssertEqual(board.chainHead[first], board.chainHead[second])
        XCTAssertEqual(board.getChainSize(first), 2)
        XCTAssertEqual(board.getNumLiberties(first), 6)

        for (x, y) in [(0, 1), (1, 0), (1, 2), (3, 1), (2, 0), (2, 2)] {
            XCTAssertTrue(board.playMove(board.getLoc(x, y), .white, false))
        }
        XCTAssertEqual(board.colors[first], .empty)
        XCTAssertEqual(board.colors[second], .empty)
        XCTAssertEqual(board.numBlackCaptures, 2)
        XCTAssertNoThrow(try board.checkConsistency())
    }

    func testMultiStoneSuicidePolicy() {
        let board = Board(3, 3)
        XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(1, 0), .black))
        for (x, y) in [(0, 0), (2, 0), (0, 1), (2, 1), (1, 2)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .white))
        }
        let suicide = board.getLoc(1, 1)
        XCTAssertTrue(board.isSuicide(suicide, .black))
        XCTAssertFalse(board.isLegal(suicide, .black, false))
        XCTAssertTrue(board.isLegal(suicide, .black, true))
        XCTAssertTrue(board.playMove(suicide, .black, true))
        XCTAssertEqual(board.colors[suicide], .empty)
        XCTAssertEqual(board.colors[board.getLoc(1, 0)], .empty)
        XCTAssertEqual(board.numBlackCaptures, 2)
        XCTAssertNoThrow(try board.checkConsistency())
    }

    func testSimpleKoSetAndCleared() {
        let board = Board(5, 5)
        for (x, y) in [(1, 2), (2, 1), (2, 3)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .black))
        }
        for (x, y) in [(2, 2), (3, 1), (4, 2), (3, 3)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .white))
        }
        let capture = board.getLoc(3, 2)
        let ko = board.getLoc(2, 2)
        XCTAssertTrue(board.playMove(capture, .black, false))
        XCTAssertEqual(board.koLoc, ko)
        XCTAssertFalse(board.isLegal(ko, .white, false))
        XCTAssertTrue(board.playMove(Board.passLoc, .white, false))
        XCTAssertEqual(board.koLoc, Board.nullLoc)
        XCTAssertTrue(board.isLegal(ko, .white, false))
        XCTAssertNoThrow(try board.checkConsistency())
    }

    func testPassAliveAndIndependentLifeArea() {
        let board = Board(7, 7)
        for (x, y) in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .black))
        }
        for (x, y) in [(5, 6), (6, 5), (5, 5)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .white))
        }

        let area = board.calculateArea()
        XCTAssertEqual(area[board.getLoc(0, 0)], .black)
        XCTAssertEqual(area[board.getLoc(2, 0)], .black)
        XCTAssertEqual(area[board.getLoc(1, 1)], .black)
        XCTAssertEqual(area[board.getLoc(6, 6)], .empty)
        XCTAssertEqual(area[board.getLoc(5, 5)], .empty)

        let independentBoard = Board(7, 7)
        for (x, y) in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)] {
            XCTAssertTrue(independentBoard.setStoneFailIfNoLibs(independentBoard.getLoc(x, y), .black))
        }
        let independent = independentBoard.calculateIndependentLifeArea()
        XCTAssertEqual(independent.area[independentBoard.getLoc(0, 0)], .black)
        XCTAssertEqual(independent.whiteMinusBlackIndependentLifeRegionCount, -1)
        XCTAssertNoThrow(try board.checkConsistency())
        XCTAssertNoThrow(try independentBoard.checkConsistency())
    }

    func testZobristHashParity() {
        let board = Board(19, 19)
        XCTAssertEqual(board.posHash.description, "CDCBC1F514D7E680FACD226074256633")
        XCTAssertTrue(board.playMove(board.getLoc(3, 3), .black, false))
        XCTAssertTrue(board.playMove(board.getLoc(16, 16), .white, false))
        XCTAssertTrue(board.playMove(board.getLoc(15, 3), .black, false))
        XCTAssertTrue(board.playMove(board.getLoc(3, 15), .white, false))
        XCTAssertEqual(board.posHash.description, "1462F357730D94C084C6AB22D7EBB63B")
        XCTAssertNoThrow(try board.checkConsistency())
    }

    func testCopyAndRecordedUndo() {
        let board = Board(9, 9)
        let initialHash = board.posHash
        let move = board.getLoc(4, 4)
        let record = board.playMoveRecorded(move, .black)
        let copied = board.copy()
        XCTAssertEqual(copied.posHash, board.posHash)
        XCTAssertTrue(copied.playMove(copied.getLoc(3, 4), .white, false))
        XCTAssertNotEqual(copied.posHash, board.posHash)
        board.undo(record)
        XCTAssertEqual(board.posHash, initialHash)
        XCTAssertEqual(board.colors[move], .empty)
        XCTAssertNoThrow(try board.checkConsistency())
    }
}
