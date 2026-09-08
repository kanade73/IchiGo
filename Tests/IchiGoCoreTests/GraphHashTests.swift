@testable import IchiGoCore
import XCTest

/// Pins the node-identity hash the DAG search depends on (see "Graph search" in
/// `docs/design-notes.md`). GraphHash is a pure function of (BoardHistory, Board, Player,
/// rules/komi) plus an explicit `prev` chain link, so every fixture below either drives real
/// `Board`/`BoardHistory` state through the public API, or calls `GraphHash.graphHash` directly
/// to isolate the reset/chain branch.
final class GraphHashTests: XCTestCase {
    private let repBound = 11
    private let drawEquiv = 0.5

    // MARK: - T-H1: pure transposition merges under the global reset condition

    func testTH1PureTranspositionMergesUnderGlobalReset() {
        /// Two independent (non-interacting) black/white exchanges, played in opposite block order.
        /// Every move here lands in a wide-open area of the board, so every graphHash step resets to
        /// a pure stateHash (graphhash.cpp:27-28) — meaning the final hash can only depend on the
        /// resulting position, not on which exchange was played first.
        func play(order: [(x: Int, y: Int, pla: Player)]) -> (hash: Hash128, board: Board) {
            let board = Board(9, 9)
            let history = BoardHistory(board, pla: .black, rules: Rules())
            for move in order {
                let loc = board.getLoc(move.x, move.y)
                XCTAssertTrue(history.isLegal(board, loc, move.pla))
                history.makeBoardMoveAssumeLegal(board, loc, move.pla)
            }
            let hash = GraphHash.fromScratch(
                hist: history,
                board: board,
                nextPla: .black,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquiv
            )
            return (hash, board)
        }

        let blockA: [(x: Int, y: Int, pla: Player)] = [(1, 1, .black), (2, 2, .white)]
        let blockB: [(x: Int, y: Int, pla: Player)] = [(6, 6, .black), (7, 7, .white)]

        let ab = play(order: blockA + blockB)
        let ba = play(order: blockB + blockA)

        XCTAssertEqual(ab.board.posHash, ba.board.posHash, "fixture sanity: both orders reach the same board")
        XCTAssertEqual(ab.hash, ba.hash)
    }

    // MARK: - T-H2: simple-ko presence differentiates stateHash

    func testTH2SimpleKoLocDifferentiatesStateHash() {
        let board = koFrameBoard()
        let captureLoc = board.getLoc(3, 2)
        XCTAssertTrue(board.playMove(captureLoc, .black, false))
        XCTAssertNotEqual(board.koLoc, Board.nullLoc, "fixture sanity: capture must set a simple-ko point")

        let history = BoardHistory(board, pla: .white, rules: Rules())
        let withKo = GraphHash.stateHash(history, board, .white, drawEquivWinsForWhite: drawEquiv)

        let noKoBoard = board.copy()
        noKoBoard.clearSimpleKoLoc()
        let withoutKo = GraphHash.stateHash(history, noKoBoard, .white, drawEquivWinsForWhite: drawEquiv)

        XCTAssertNotEqual(withKo, withoutKo)
    }

    // MARK: - T-H3: chain does not merge across different path lengths in a small (ko) region

    func testTH3ChainDoesNotMergeAcrossDifferentPathsInSmallRegion() {
        /// Two paths reach the byte-identical final position (same stateHash) via a different number
        /// of preliminary pass-pairs, both finishing with the same small-region ko capture. Passes
        /// always take graphHash's non-reset branch (simpleRepetitionBoundGt(passLoc, _) is false by
        /// construction — T-H6d), so the accumulated chain differs between the two paths even though
        /// the position they land on does not — the final (non-resetting) capture step must therefore
        /// disagree too.
        func replay(passPairs: Int) -> (hash: Hash128, board: Board, history: BoardHistory) {
            let board = koFrameBoard()
            let history = BoardHistory(board, pla: .black, rules: Rules())
            for _ in 0 ..< passPairs {
                XCTAssertTrue(history.isLegal(board, Board.passLoc, .black))
                history.makeBoardMoveAssumeLegal(board, Board.passLoc, .black)
                XCTAssertTrue(history.isLegal(board, Board.passLoc, .white))
                history.makeBoardMoveAssumeLegal(board, Board.passLoc, .white)
            }
            let captureLoc = board.getLoc(3, 2)
            XCTAssertTrue(history.isLegal(board, captureLoc, .black))
            history.makeBoardMoveAssumeLegal(board, captureLoc, .black)
            let hash = GraphHash.fromScratch(
                hist: history,
                board: board,
                nextPla: .white,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquiv
            )
            return (hash, board, history)
        }

        let short = replay(passPairs: 1)
        let long = replay(passPairs: 2)

        XCTAssertEqual(short.board.posHash, long.board.posHash, "fixture sanity: same final position")
        XCTAssertEqual(short.board.koLoc, long.board.koLoc)
        XCTAssertEqual(
            GraphHash.stateHash(short.history, short.board, .white, drawEquivWinsForWhite: drawEquiv),
            GraphHash.stateHash(long.history, long.board, .white, drawEquivWinsForWhite: drawEquiv),
            "fixture sanity: identical stateHash despite different path lengths"
        )

        XCTAssertNotEqual(short.hash, long.hash)
    }

    // MARK: - T-H4: pass-state (consecutiveEndingPasses / passWouldEndPhase / isGameFinished)

    func testTH4PassStateFoldsIntoStateHash() {
        let board = Board(5, 5)
        let history = BoardHistory(board, pla: .black, rules: Rules())

        XCTAssertEqual(history.consecutiveEndingPasses, 0)
        XCTAssertFalse(history.passWouldEndPhase(board, .black))
        XCTAssertFalse(history.isGameFinished)
        let hash0 = GraphHash.stateHash(history, board, .black, drawEquivWinsForWhite: drawEquiv)

        XCTAssertTrue(history.isLegal(board, Board.passLoc, .black))
        history.makeBoardMoveAssumeLegal(board, Board.passLoc, .black)
        XCTAssertEqual(history.consecutiveEndingPasses, 1)
        XCTAssertTrue(history.passWouldEndPhase(board, .white))
        XCTAssertFalse(history.isGameFinished)
        let hash1 = GraphHash.stateHash(history, board, .white, drawEquivWinsForWhite: drawEquiv)

        XCTAssertTrue(history.isLegal(board, Board.passLoc, .white))
        history.makeBoardMoveAssumeLegal(board, Board.passLoc, .white)
        XCTAssertEqual(history.consecutiveEndingPasses, 2)
        XCTAssertTrue(history.isGameFinished, "fixture sanity: area rules score+end on 2 consecutive passes")
        let hash2 = GraphHash.stateHash(history, board, .black, drawEquivWinsForWhite: drawEquiv)

        XCTAssertNotEqual(hash0, hash1)
        XCTAssertNotEqual(hash1, hash2)
        XCTAssertNotEqual(hash0, hash2)
    }

    // MARK: - T-H5: fromScratch reproduces the same values as incremental chaining, at every prefix

    func testTH5FromScratchMatchesIncrementalChainAtEveryPrefix() {
        // A deterministic stand-in for "replay N self-play moves": mixes wide-open placements
        // (graphHash resets) with passes (graphHash never resets on a pass — T-H6d), so both
        // branches of graphHash's reset/chain decision are exercised along the same path.
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules())
        let moves: [(loc: Loc, pla: Player)] = [
            (board.getLoc(2, 2), .black),
            (board.getLoc(6, 6), .white),
            (Board.passLoc, .black),
            (Board.passLoc, .white),
            (board.getLoc(2, 6), .black),
            (board.getLoc(6, 2), .white),
            (Board.passLoc, .black),
            (board.getLoc(4, 4), .white),
            (board.getLoc(7, 7), .black),
            (Board.passLoc, .white),
        ]

        var chain = Hash128()
        var prevMoveLoc = Board.nullLoc

        for move in moves {
            // Layer for the state as of *before* this move, viewed with mover = this move's player —
            // matches one loop iteration of GraphHash.fromScratch.
            chain = GraphHash.graphHash(
                prev: chain,
                hist: history,
                board: board,
                prevMoveLoc: prevMoveLoc,
                nextPla: move.pla,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquiv
            )
            XCTAssertTrue(history.isLegal(board, move.loc, move.pla))
            history.makeBoardMoveAssumeLegal(board, move.loc, move.pla)
            prevMoveLoc = move.loc

            let nextPla = history.presumedNextMovePla
            let incremental = GraphHash.graphHash(
                prev: chain,
                hist: history,
                board: board,
                prevMoveLoc: prevMoveLoc,
                nextPla: nextPla,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquiv
            )
            let fromScratch = GraphHash.fromScratch(
                hist: history,
                board: board,
                nextPla: nextPla,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquiv
            )
            XCTAssertEqual(incremental, fromScratch, "mismatch after move to \(move.loc)")
        }
    }

    // MARK: - T-H6: simpleRepetitionBoundGt (existing primitive — regression pin, not a re-port)

    func testTH6ASingleStoneKoShapeIsSmallRegion() {
        let board = koFrameBoard()
        let captureLoc = board.getLoc(3, 2)
        XCTAssertTrue(board.playMove(captureLoc, .black, false))
        XCTAssertFalse(board.simpleRepetitionBoundGt(captureLoc, repBound))
    }

    func testTH6BMoveIntoBigOpenAreaIsLargeRegion() {
        let board = Board(9, 9)
        let loc = board.getLoc(4, 4)
        XCTAssertTrue(board.playMove(loc, .black, false))
        XCTAssertTrue(board.simpleRepetitionBoundGt(loc, repBound))
    }

    func testTH6CFastQuitoutFromChainSizePlusLiberties() {
        let board = Board(9, 9)
        var last = Board.nullLoc
        for x in 1 ... 6 {
            last = board.getLoc(x, 4)
            XCTAssertTrue(board.setStoneFailIfNoLibs(last, .black))
        }
        // A 6-stone open line has chain size 6 and 14 liberties (2 per stone + 2 ends): 20 > 11,
        // tripping the fast quitout before the empty-region BFS ever runs.
        XCTAssertTrue(board.simpleRepetitionBoundGt(last, repBound))
    }

    func testTH6DPassAndNullLocAreAlwaysSmallRegion() {
        let board = Board(9, 9)
        XCTAssertFalse(board.simpleRepetitionBoundGt(Board.passLoc, repBound))
        XCTAssertFalse(board.simpleRepetitionBoundGt(Board.nullLoc, repBound))
    }

    // MARK: - T-H7: rules/komi are folded into stateHash

    func testTH7RulesAndKomiChangeStateHash() {
        let board = Board(5, 5)

        func hash(_ rules: Rules) -> Hash128 {
            let history = BoardHistory(board, pla: .black, rules: rules)
            return GraphHash.stateHash(history, board, .black, drawEquivWinsForWhite: drawEquiv)
        }

        let baseline = Rules()
        let baselineHash = hash(baseline)

        var komiVaried = baseline
        komiVaried.komi += 1
        XCTAssertNotEqual(hash(komiVaried), baselineHash, "komi")

        var koRuleVaried = baseline
        koRuleVaried.koRule = .simple
        XCTAssertNotEqual(hash(koRuleVaried), baselineHash, "koRule")

        var scoringVaried = baseline
        scoringVaried.scoringRule = .territory
        XCTAssertNotEqual(hash(scoringVaried), baselineHash, "scoringRule")

        var taxVaried = baseline
        taxVaried.taxRule = .seki
        XCTAssertNotEqual(hash(taxVaried), baselineHash, "taxRule")

        var suicideVaried = baseline
        suicideVaried.multiStoneSuicideLegal.toggle()
        XCTAssertNotEqual(hash(suicideVaried), baselineHash, "multiStoneSuicideLegal")

        var buttonVaried = baseline
        buttonVaried.hasButton = true
        XCTAssertNotEqual(hash(buttonVaried), baselineHash, "hasButton")

        var friendlyVaried = baseline
        friendlyVaried.friendlyPassOk = true
        XCTAssertNotEqual(hash(friendlyVaried), baselineHash, "friendlyPassOk")
    }

    // MARK: - Shared fixtures

    /// 5x5 board with a one-move-from-capturing simple-ko shape: black plays `(3,2)` next to capture
    /// the lone white stone at `(2,2)`, which then becomes the ko point. Matches the shape already
    /// used by `BoardTests.testSimpleKoSetAndCleared`.
    private func koFrameBoard() -> Board {
        let board = Board(5, 5)
        for (x, y) in [(1, 2), (2, 1), (2, 3)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .black))
        }
        for (x, y) in [(2, 2), (3, 1), (4, 2), (3, 3)] {
            XCTAssertTrue(board.setStoneFailIfNoLibs(board.getLoc(x, y), .white))
        }
        return board
    }
}
