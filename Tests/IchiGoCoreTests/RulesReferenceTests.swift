import IchiGoCore
import XCTest

final class RulesReferenceTests: XCTestCase {
    func testRulesRoundTrips() throws {
        let rules = [
            Rules(koRule: .simple, multiStoneSuicideLegal: false, komi: 1),
            Rules(koRule: .positional, multiStoneSuicideLegal: false, hasButton: true, friendlyPassOk: true, komi: 3),
            Rules(koRule: .situational, multiStoneSuicideLegal: true, komi: 5.5),
            Rules(koRule: .simple, taxRule: .seki, multiStoneSuicideLegal: false, komi: 1),
            Rules(koRule: .positional, taxRule: .seki, whiteHandicapBonusRule: .n, friendlyPassOk: true, komi: 4.5),
            Rules(
                koRule: .situational,
                scoringRule: .territory,
                multiStoneSuicideLegal: true,
                hasButton: true,
                whiteHandicapBonusRule: .n,
                friendlyPassOk: true,
                komi: 6.5
            ),
            Rules(
                koRule: .simple,
                scoringRule: .territory,
                taxRule: .seki,
                multiStoneSuicideLegal: false,
                hasButton: true,
                komi: 2
            ),
            Rules(
                koRule: .situational,
                scoringRule: .territory,
                taxRule: .seki,
                hasButton: true,
                friendlyPassOk: true,
                komi: 5.5
            ),
            Rules(koRule: .simple, taxRule: .all, hasButton: false, whiteHandicapBonusRule: .nMinusOne, komi: 2.5),
            Rules(
                koRule: .positional,
                taxRule: .all,
                multiStoneSuicideLegal: false,
                hasButton: true,
                friendlyPassOk: true,
                komi: 3
            ),
            Rules(
                koRule: .positional,
                scoringRule: .territory,
                taxRule: .all,
                multiStoneSuicideLegal: false,
                hasButton: true,
                whiteHandicapBonusRule: .nMinusOne,
                komi: 4
            ),
            Rules(koRule: .situational, scoringRule: .territory, taxRule: .all, whiteHandicapBonusRule: .n, komi: 6.5),
        ]
        for rule in rules {
            XCTAssertEqual(try Rules.parseRules(rule.toString()), rule)
            XCTAssertEqual(try Rules.parseRulesWithoutKomi(rule.toStringNoKomi(), komi: rule.komi), rule)
            XCTAssertEqual(try Rules.parseRules(rule.toJsonString()), rule)
            XCTAssertEqual(try Rules.parseRulesWithoutKomi(rule.toJsonStringNoKomi(), komi: rule.komi), rule)
            XCTAssertEqual(
                try Rules.parseRulesWithoutKomi(rule.toJsonStringNoKomiMaybeOmitStuff(), komi: rule.komi),
                rule
            )
        }
    }

    func testRulesParsingBug() throws {
        let parsed = try Rules.parseRules("komi23taxALL")
        let actual = "\n\(parsed)\n"
        let expected = #"""

        koPOSITIONALscoreAREAtaxALLsui1komi23
        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testAreaRules() throws {
        let board = try Board.parseBoard(4, 4, "....\n....\n....\n....")
        let rules = Rules(koRule: .positional, komi: 0.5)
        let history = BoardHistory(board, pla: .black, rules: rules)
        try play(history, board, 1, 1, .black)
        try play(history, board, 2, 2, .white)
        try play(history, board, 1, 2, .black)
        try play(history, board, 2, 1, .white)
        try play(history, board, 1, 3, .black)
        try play(history, board, 2, 3, .white)
        try play(history, board, 1, 0, .black)
        try play(history, board, 2, 0, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertTrue(history.isGameFinished)
        XCTAssertEqual(history.winner, .white)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, 0.5)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, 3, 2, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, 0.5)
        let expected = #"""

        HASH: C5B7EC66E875EF237ADC456CEE8436EA
           A B C D
         4 . X O .
         3 . X O .
         2 . X O O
         1 . X O .
        """#
        BoardReferenceTestSupport.assertCppReference("\n\(board)", expected)
    }

    func testTerritoryRules() throws {
        let board = try Board.parseBoard(4, 4, "....\n....\n....\n....")
        let rules = Rules(
            koRule: .positional,
            scoringRule: .territory,
            taxRule: .seki,
            multiStoneSuicideLegal: true,
            komi: 0.5
        )
        let history = BoardHistory(board, pla: .black, rules: rules)
        let moves: [(Int, Int, Player)] = [
            (1, 1, .black), (2, 2, .white), (1, 2, .black), (2, 1, .white),
            (1, 3, .black), (2, 3, .white), (1, 0, .black), (2, 0, .white),
        ]
        for (x, y, pla) in moves {
            try play(history, board, x, y, pla)
        }
        try play(history, board, Board.passLoc, .black)
        try play(history, board, 3, 2, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.encorePhase, 1)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.encorePhase, 2)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, -0.5)
        var actual = "\n\(board)\n"
        try play(history, board, 3, 1, .black)
        try play(history, board, Board.passLoc, .white)
        try play(history, board, Board.passLoc, .black)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, -0.5)
        actual += "\(board)\n"
        try play(history, board, 0, 1, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, 3.5)
        actual += "\(board)\n"
        try play(history, board, 0, 2, .black)
        try play(history, board, 3, 0, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, -0.5)
        actual += "\(board)"
        let expected = #"""

        HASH: C5B7EC66E875EF237ADC456CEE8436EA
           A B C D
         4 . X O .
         3 . X O .
         2 . X O O
         1 . X O .


        HASH: 8DC655E188DC780F2DC52C2AB2466765
           A B C D
         4 . X O .
         3 . X O X
         2 . X O O
         1 . X O .


        HASH: 6DC344129837C704ECC19ED88E1028E9
           A B C D
         4 . X O .
         3 O X O X
         2 . X O O
         1 . X O .


        HASH: 3E079ADF838F23351E0B68CCA0928806
           A B C D
         4 . X O O
         3 O X O .
         2 X X O O
         1 . X O .
        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testSimpleKoRules() throws {
        let (board, rules) = try koTestSetup(koRule: .simple)
        let history = BoardHistory(board, pla: .black, rules: rules)
        var actual = "\n"
        try play(history, board, 5, 1, .black)
        actual += "After black ko capture:\n" + printIllegalMoves(board, history, .white)
        try play(history, board, Board.passLoc, .white)
        actual += "After black ko capture and one pass:\n" + printIllegalMoves(board, history, .black)
        XCTAssertTrue(history.passWouldEndPhase(board, .black))
        XCTAssertFalse(history.passWouldEndGame(board, .black))
        try play(history, board, 2, 3, .black)
        actual += "After black ko capture and one pass and black other move:\n"
            + printIllegalMoves(board, history, .white)
        try play(history, board, 5, 0, .white)
        actual += "White recapture:\n" + printIllegalMoves(board, history, .black)
        try play(history, board, 3, 2, .black)
        actual += "Beginning sending two returning one cycle\n"
        try play(history, board, 2, 0, .white)
        actual += printIllegalMoves(board, history, .black)
        try play(history, board, 0, 0, .black)
        actual += printIllegalMoves(board, history, .white)
        try play(history, board, 1, 0, .white)
        actual += printIllegalMoves(board, history, .black)
        try play(history, board, Board.passLoc, .black)
        actual += printIllegalMoves(board, history, .white)
        try play(history, board, 2, 0, .white)
        actual += printIllegalMoves(board, history, .black)
        try play(history, board, 0, 0, .black)
        actual += printIllegalMoves(board, history, .white)
        try play(history, board, 1, 0, .white)
        actual += printIllegalMoves(board, history, .black)
        try play(history, board, Board.passLoc, .black)
        actual += printIllegalMoves(board, history, .white)
        XCTAssertEqual(history.encorePhase, 1)
        try play(history, board, Board.passLoc, .white)
        try play(history, board, Board.passLoc, .black)
        try play(history, board, Board.passLoc, .white)
        try play(history, board, Board.passLoc, .black)
        XCTAssertEqual(history.encorePhase, 2)
        actual += gameResult(history)
        let expected = #"""

        After black ko capture:
        Illegal: (5,0) O
        After black ko capture and one pass:
        After black ko capture and one pass and black other move:
        White recapture:
        Illegal: (5,1) X
        Beginning sending two returning one cycle
        Winner: Black
        W-B Score: -1.5
        isNoResult: 0
        isResignation: 0
        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testPositionalKoRules() throws {
        let (board, rules) = try koTestSetup(koRule: .positional)
        let history = BoardHistory(board, pla: .black, rules: rules)
        var actual = "\n"
        try play(history, board, 5, 1, .black)
        actual += "After black ko capture:\n" + printIllegalMoves(board, history, .white)
        try play(history, board, Board.passLoc, .white)
        actual += "After black ko capture and one pass:\n" + printIllegalMoves(board, history, .black)
        let temporaryBoard = board.copy()
        let temporaryHistory = history.copy()
        try play(temporaryHistory, temporaryBoard, Board.passLoc, .black)
        XCTAssertEqual(temporaryHistory.encorePhase, 1)
        try play(history, board, 3, 2, .black)
        actual += "Beginning sending two returning one cycle\n"
        try play(history, board, 2, 0, .white)
        actual += "After white sends two?\n" + printIllegalMoves(board, history, .black)
        try play(history, board, 0, 0, .black)
        actual += "Can white recapture?\n" + printIllegalMoves(board, history, .white)
        try play(history, board, 5, 0, .white)
        actual += "After white recaptures the other ko instead\n" + printIllegalMoves(board, history, .black)
        try play(history, board, Board.passLoc, .black)
        actual += "After white recaptures the other ko instead and black passes\n"
            + printIllegalMoves(board, history, .white)
        try play(history, board, 1, 0, .white)
        actual += "After white now returns 1\n" + printIllegalMoves(board, history, .black)
        try play(history, board, Board.passLoc, .black)
        actual += "After white now returns 1 and black passes\n" + printIllegalMoves(board, history, .white)
        try play(history, board, 2, 0, .white)
        actual += "After white sends 2 again\n" + printIllegalMoves(board, history, .black)
        let illegal = board.getLoc(0, 0)
        XCTAssertFalse(history.isLegal(board, illegal, .black))
        XCTAssertTrue(history.isLegalTolerant(board, illegal, .black))
        history.makeBoardMoveAssumeLegal(board, illegal, .black)
        XCTAssertEqual(history.numConsecValidTurnsThisGame, 0)
        let expected = #"""

        After black ko capture:
        Illegal: (5,0) O
        After black ko capture and one pass:
        Beginning sending two returning one cycle
        After white sends two?
        Can white recapture?
        Illegal: (1,0) O
        After white recaptures the other ko instead
        Illegal: (5,1) X
        After white recaptures the other ko instead and black passes
        After white now returns 1
        Illegal: (5,1) X
        After white now returns 1 and black passes
        After white sends 2 again
        Illegal: (0,0) X
        Illegal: (5,1) X
        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    private func koTestSetup(koRule: Rules.KoRule) throws -> (Board, Rules) {
        let board = try Board.parseBoard(6, 5, ".o.xxo\noxxxo.\no.x.oo\nxx.oo.\noooo.o")
        let rules = Rules(
            koRule: koRule,
            scoringRule: .territory,
            taxRule: .seki,
            multiStoneSuicideLegal: false,
            komi: 0.5
        )
        return (board, rules)
    }

    private func printIllegalMoves(_ board: Board, _ history: BoardHistory, _ pla: Player) -> String {
        var output = ""
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                let loc = board.getLoc(x, y)
                let isPseudolegalEmpty = board.colors[loc] == .empty
                    && !board.isIllegalSuicide(loc, pla, history.rules.multiStoneSuicideLegal)
                if isPseudolegalEmpty, !history.isLegal(board, loc, pla) {
                    output += "Illegal: \(Location.toStringMach(loc, board.xSize)) \(pla == .black ? "X" : "O")\n"
                }
            }
        }
        return output
    }

    private func gameResult(_ history: BoardHistory) -> String {
        let winner = history.winner == .black ? "Black" : history.winner == .white ? "White" : "Empty"
        return "Winner: \(winner)\nW-B Score: \(history.finalWhiteMinusBlackScore)\n"
            + "isNoResult: \(history.isNoResult ? 1 : 0)\nisResignation: \(history.isResignation ? 1 : 0)\n"
    }

    private func play(_ history: BoardHistory, _ board: Board, _ x: Int, _ y: Int, _ pla: Player) throws {
        try play(history, board, board.getLoc(x, y), pla)
    }

    private func play(_ history: BoardHistory, _ board: Board, _ loc: Loc, _ pla: Player) throws {
        guard history.isLegal(board, loc, pla) else { throw TestError.illegalMove(loc, pla) }
        history.makeBoardMoveAssumeLegal(board, loc, pla)
    }

    private enum TestError: Error { case illegalMove(Loc, Player) }
}
