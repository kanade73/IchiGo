import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import LogicModel
import XCTest

/// Fake evaluator (Tests only): policy concentrated on chosen indices, fixed expected result.
actor FakeEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    var expected: Float
    var favoured: [Int]
    var calls = 0
    var failAfter: Int? = nil
    var batchSizes: [Int] = []

    init(sizes: Set<Int>, expected: Float = 0.6, favoured: [Int] = []) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
        self.expected = expected
        self.favoured = favoured
    }

    func setFailAfter(_ n: Int?) { failAfter = n }

    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        calls += 1
        batchSizes.append(positions.count)
        if let f = failAfter, calls > f { throw SearchError(message: "fake failure") }
        return positions.map { s in
            let P = s.boardSize * s.boardSize
            var policy = [Float](repeating: 0, count: P + 1)
            var mass: Float = 0
            for i in 0 ... P where s.legal[i] == 1 { policy[i] = favoured.contains(i) ? 10 : 1; mass += policy[i] }
            policy = policy.map { $0 / mass }
            return LogicEvaluation(policy: policy, winDrawLoss: [expected, 0, 1 - expected], expectedResult: expected, scoreMean: 2, ownership: [Float](repeating: 0, count: P))
        }
    }

    func preWarm(size: Int) async throws {}
}

/// Fake evaluator (Tests only) for the `ValueSource.ownership` search test: `winDrawLoss`/
/// `expectedResult` are always uninformative (0.5, so `.network` mode never distinguishes any
/// move), but the ownership head strongly favours whichever child follows `favouredMove` (played
/// from the root) — the position `.ownership` mode must find and `.network` mode must not.
actor OwnershipFakeEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    let favouredMove: MoveCoord
    init(sizes: Set<Int>, favouredMove: MoveCoord) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
        self.favouredMove = favouredMove
    }

    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        positions.map { s in
            let P = s.boardSize * s.boardSize
            var policy = [Float](repeating: 0, count: P + 1)
            var mass: Float = 0
            for i in 0 ... P where s.legal[i] == 1 { policy[i] = 1; mass += 1 }
            policy = policy.map { $0 / mass }
            // Only the child reached by playing `favouredMove` at the root gets a strongly
            // negative ownership sum (in the *child's* to-move perspective) — i.e. a strong
            // advantage for whoever just moved there.
            let isFavouredChild = s.recentMoves.first == favouredMove
            let own: Float = isFavouredChild ? -1 : 0
            return LogicEvaluation(policy: policy, winDrawLoss: [0.5, 0, 0.5], expectedResult: 0.5, scoreMean: 0, ownership: [Float](repeating: own, count: P))
        }
    }

    func preWarm(size: Int) async throws {}
}

/// Fake evaluator (Tests only) for `ValueSource.rollout`: `winDrawLoss`/`expectedResult` are fixed
/// and uninformative (0.5), independent of the position, so any rollout-driven behaviour cannot be
/// coming from the network baseline; `policy` always puts full probability mass on pass. Pass is
/// legal essentially everywhere, so this makes rollout playouts fully deterministic (no randomness
/// ever actually gets sampled, regardless of `SearchSettings.rolloutRNGSeed`) — every playout ends
/// after at most two plies (a pass by each side), letting `value_rollout` be computed by hand from
/// the position's exact area score.
actor PassOnlyEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    let expected: Float
    init(sizes: Set<Int>, expected: Float = 0.5) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
        self.expected = expected
    }

    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        positions.map { s in
            let P = s.boardSize * s.boardSize
            var policy = [Float](repeating: 0, count: P + 1)
            policy[P] = 1
            return LogicEvaluation(policy: policy, winDrawLoss: [expected, 0, 1 - expected], expectedResult: expected, scoreMean: 0, ownership: [Float](repeating: 0, count: P))
        }
    }

    func preWarm(size: Int) async throws {}
}

/// Fake evaluator (Tests only) for `testSearchUnderRolloutFindsWinningMoveInCaptureRace`:
/// `winDrawLoss`/`expectedResult` are fixed and uninformative (0.5), so `.network` mode could never
/// prefer the capturing move. `policy` at the *root* (`moveNumber == 0`) is uniform over every
/// legal move, so PUCT's exploration term visits every root candidate; at every position reached
/// after at least one move (`moveNumber >= 1` — every rollout playout step, and every leaf's own
/// expansion), it is 100% mass on pass (always legal), which forces every rollout playout to end
/// immediately (two passes) and score the board exactly as it stands right after the root's single
/// move — deterministic, no RNG dependence at all, so `value_rollout` differs between moves purely
/// from the resulting board's exact area score.
actor RootUniformThenPassEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    init(sizes: Set<Int>) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
    }

    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        positions.map { s in
            let P = s.boardSize * s.boardSize
            var policy = [Float](repeating: 0, count: P + 1)
            if s.moveNumber == 0 {
                var mass: Float = 0
                for i in 0 ... P where s.legal[i] == 1 { policy[i] = 1; mass += 1 }
                policy = policy.map { $0 / mass }
            } else {
                policy[P] = 1
            }
            return LogicEvaluation(policy: policy, winDrawLoss: [0.5, 0, 0.5], expectedResult: 0.5, scoreMean: 0, ownership: [Float](repeating: 0, count: P))
        }
    }

    func preWarm(size: Int) async throws {}
}

final class SearchTests: XCTestCase {
    /// Opt-in performance smoke test for comparing tree overhead across board sizes. Run with
    /// `ICHIGO_RUN_BENCHMARK=1 swift test --filter testSearchMicroBenchmarkVisitsPerSecond`.
    func testSearchMicroBenchmarkVisitsPerSecond() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ICHIGO_RUN_BENCHMARK"] == "1")
        for size in [9, 19] {
            let ev = FakeEvaluator(sizes: [size], expected: 0.5)
            var settings = SearchSettings()
            settings.leafBatch = 8
            let game = try GameState(boardSize: size, komi: size == 9 ? 7 : 7.5)
            let search = try Search(evaluator: ev, modelHash: "benchmark", settings: settings, initial: game.record)
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try await search.run(visits: 400)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
            print("SEARCH_BENCH size=\(size) visits=\(result.rootVisits) elapsed=\(String(format: "%.6f", elapsed)) visitsPerSecond=\(String(format: "%.1f", Double(result.rootVisits) / elapsed))")
            XCTAssertEqual(result.rootVisits, 400)
        }
    }

    func testHandComputedSelectionAndBackup() async throws {
        // Root: black to move, NN expected 0.6 (black) => white value -0.2. Favoured moves 0 and 1 (prior 10/…).
        // With leafBatch 1 and 3 visits: root eval (1 visit), then two leaf expansions picked by PUCT.
        let ev = FakeEvaluator(sizes: [9], expected: 0.6, favoured: [0, 1])
        var st = SearchSettings(); st.leafBatch = 1
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 3)
        XCTAssertEqual(r.rootVisits, 3)
        // Both favoured moves have equal prior; ties → lower index first, then second visit goes to index 1
        // because after one visit at index 0: Q(0) = -Qwhite... children evaluated with same expected 0.6 for the
        // side to move (white at depth 1) => white value +0.2, parent(black) view -0.2 vs unvisited FPU = parent's
        // own view of NN value (+0.2). So index 1 (unvisited) wins the second visit.
        let top = r.candidates.filter { $0.visits > 0 }.map(\.index).sorted()
        XCTAssertEqual(top, [0, 1])
        XCTAssertEqual(r.move, .point(x: 0, y: 0))  // tie on visits → prior tie → index 0
        // search expected for black: mean white value over 3 visits: (-0.2 + 0.2 + 0.2)/3 = 0.0667 → expected 0.4667
        XCTAssertEqual(r.searchExpected, (1 - 0.0667) / 2, accuracy: 1e-3)
        XCTAssertEqual(r.rootRawExpected, 0.6, accuracy: 1e-6)
        let calls = await ev.calls
        XCTAssertEqual(calls, 3)
    }

    func testDrawTerminalIsHalfAndTerminalNeverCallsNN() async throws {
        // Empty board, komi 0: two passes end the game with a draw (score 0).
        let ev = FakeEvaluator(sizes: [9], expected: 0.9, favoured: [81])
        var st = SearchSettings(); st.leafBatch = 1
        let g = try GameState(boardSize: 9, komi: 0)
        try g.play(.black, .pass)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 4)
        let passCand = r.candidates.first { $0.move == .pass }!
        XCTAssertGreaterThan(passCand.visits, 0)
        XCTAssertEqual(passCand.expectedScore, 0.5, accuracy: 1e-9)   // exact draw
        XCTAssertEqual(passCand.scoreLead, 0, accuracy: 1e-9)
        let calls = await ev.calls
        XCTAssertEqual(calls, 1 + (r.rootVisits - 1 - passCand.visits))  // terminal visits made no NN call
    }

    func testOnlyLegalMovesAndCompletionBothSizes() async throws {
        for S in [9, 19] {
            let ev = FakeEvaluator(sizes: [S], expected: 0.5)
            let g = try GameState(boardSize: S, komi: 7)
            try g.play(.black, .point(x: 2, y: 2))
            let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record)
            let r = try await search.run(visits: 40)
            XCTAssertEqual(r.rootVisits, 40)
            XCTAssertEqual(r.toMove, .white)
            XCTAssertFalse(r.candidates.contains { $0.move == .point(x: 2, y: 2) })
            XCTAssertEqual(r.candidates.count, S * S)  // S*S-1 points + pass
            XCTAssertTrue(g.isLegal(.white, r.move))
            let sizes = await ev.batchSizes
            XCTAssertTrue(sizes.dropFirst().allSatisfy { $0 <= 8 })
        }
    }

    func testReservationsReleasedOnFailureAndSearchContinues() async throws {
        let ev = FakeEvaluator(sizes: [9], expected: 0.5)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record)
        await ev.setFailAfter(2)
        do { _ = try await search.run(visits: 30); XCTFail("expected failure") } catch {}
        let v1 = await search.rootVisits()
        await ev.setFailAfter(nil)
        let r = try await search.run(visits: 30)
        XCTAssertEqual(r.rootVisits, 30)
        XCTAssertGreaterThan(r.rootVisits, v1)
        // total edge visits at root == rootVisits - 1 (no double counting, no leaked reservations)
        XCTAssertEqual(r.candidates.reduce(0) { $0 + $1.visits }, r.rootVisits - 1)
    }

    func testTreeReuseAndInvalidation() async throws {
        let ev = FakeEvaluator(sizes: [9], expected: 0.5, favoured: [40])
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record)
        let r = try await search.run(visits: 50)
        let best = r.candidates[0]
        try await search.makeMove(best.move)
        let reused = await search.rootVisits()
        XCTAssertEqual(reused, best.visits)
        XCTAssertGreaterThan(reused, 0)
        let s = await search.rootRecord()
        XCTAssertEqual(s.moves.count, 1)
        try await search.reset(to: g.record)
        let after = await search.rootVisits()
        XCTAssertEqual(after, 0)
        do { try await search.makeMove(.point(x: 40, y: 0)); XCTFail() } catch {}
    }

    /// docs/spec/03-engine.md §3-4 "値ソース": with a wdl head that never distinguishes any move
    /// (always 0.5) but an ownership head that clearly favours one, `.ownership` must find and
    /// play that move while plain `.network` (which ignores ownership for the value) cannot and
    /// falls back to the lowest-index legal move (all Q tie at 0, so PUCT ties break by index).
    func testValueSourceOwnershipPicksMoveNetworkModeCannotSee() async throws {
        let favoured = MoveCoord.point(x: 5, y: 0)  // index 5 on 9x9: not the lowest legal index
        let ev = OwnershipFakeEvaluator(sizes: [9], favouredMove: favoured)
        var st = SearchSettings(); st.leafBatch = 1
        let g = try GameState(boardSize: 9, komi: 7)

        st.valueSource = .ownership(k: 6, b: 1)
        let ownershipSearch = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let ownershipResult = try await ownershipSearch.run(visits: 200)
        XCTAssertEqual(ownershipResult.move, favoured)

        st.valueSource = .network
        let networkSearch = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let networkResult = try await networkSearch.run(visits: 200)
        XCTAssertNotEqual(networkResult.move, favoured)
    }

    /// docs/implementation-status.md 2026-09-10 §4-5 "value_rollout equals the known outcome":
    /// `PassOnlyEvaluator` makes every playout deterministically double-pass, so `value_rollout`
    /// can be computed by hand from the (unchanged) board's exact area score. One black pass has
    /// already been played, so white's single forced pass reaches the natural two-pass end
    /// (`maxMovesHit == 0`) on an otherwise empty board: score == komi (7) == a certain white win.
    func testRolloutValueMatchesKnownOutcomeNaturalTermination() async throws {
        let ev = PassOnlyEvaluator(sizes: [9])
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .pass)   // white to move, board still empty
        var st = SearchSettings(); st.valueSource = .rollout(count: 5, maxMoves: 20, weightNetwork: 0)   // pure rollout value
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 1)
        XCTAssertEqual(r.searchExpected, 1.0, accuracy: 1e-9)     // white (root's to-move) wins with certainty
        XCTAssertEqual(r.rootRawExpected, 0.5, accuracy: 1e-9)    // raw network wdl is untouched by rollout
        let diag = await search.rolloutDiagnostics()
        XCTAssertEqual(diag.leaves, 1)
        XCTAssertEqual(diag.playouts, 5)
        XCTAssertEqual(diag.maxMovesHit, 0)
        XCTAssertEqual(diag.maxMovesHitFraction, 0, accuracy: 1e-9)
    }

    /// docs/implementation-status.md 2026-09-10 §4-5 "maxMoves cap honoured and flagged": with
    /// `maxMoves = 1` from a fresh (zero-pass) leaf, the rollout loop plays exactly one ply (black's
    /// forced pass) and stops before white's reply ever happens, so every playout is scored as-is
    /// (approximation) instead of via a natural two-pass end — `maxMovesHit` must be 100%. The
    /// board is unchanged (a pass doesn't touch it), so the known outcome is identical to the
    /// natural-termination test above, just reached via the cap instead of two real passes.
    func testRolloutMaxMovesCapHonouredAndFlagged() async throws {
        let ev = PassOnlyEvaluator(sizes: [9])
        let g = try GameState(boardSize: 9, komi: 7)   // fresh board, black to move, zero passes so far
        var st = SearchSettings(); st.valueSource = .rollout(count: 4, maxMoves: 1, weightNetwork: 0)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 1)
        XCTAssertEqual(r.searchExpected, 0.0, accuracy: 1e-9)   // black (root's to-move) loses with certainty
        let diag = await search.rolloutDiagnostics()
        XCTAssertEqual(diag.leaves, 1)
        XCTAssertEqual(diag.playouts, 4)
        XCTAssertEqual(diag.maxMovesHit, 4)
        XCTAssertEqual(diag.maxMovesHitFraction, 1.0, accuracy: 1e-9)
    }

    /// docs/implementation-status.md 2026-09-10 §4-5 "count/blend maths": with a non-uninformative
    /// network baseline (0.8, to-move) and `weightNetwork = 0.25`, checks
    /// `leaf value = weightNetwork * e_nn(white) + (1 - weightNetwork) * value_rollout` exactly.
    func testRolloutCountAndBlendMaths() async throws {
        let ev = PassOnlyEvaluator(sizes: [9], expected: 0.8)
        let g = try GameState(boardSize: 9, komi: 7)   // fresh board, black to move
        var st = SearchSettings(); st.valueSource = .rollout(count: 6, maxMoves: 20, weightNetwork: 0.25)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 1)
        // e_nn(white) = 1 - 0.8 = 0.2 (black to move); value_rollout (white) = 1.0 (empty board
        // scores as pure komi, white wins, identically for all 6 deterministic playouts).
        // blended = 0.25*0.2 + 0.75*1.0 = 0.8 -> white value 2*0.8-1 = 0.6 -> black's (root's
        // to-move) expected score = (-0.6+1)/2 = 0.2. (accuracy 1e-6, not 1e-9: `expected` is a
        // `Float` internally, so 0.8/0.2 pick up ~1e-7-scale Float->Double representation error.)
        XCTAssertEqual(r.searchExpected, 0.2, accuracy: 1e-6)
        XCTAssertEqual(r.rootRawExpected, 0.8, accuracy: 1e-6)
        let diag = await search.rolloutDiagnostics()
        XCTAssertEqual(diag.leaves, 1)
        XCTAssertEqual(diag.playouts, 6)
        XCTAssertEqual(diag.maxMovesHit, 0)
    }

    /// `count == 0` degenerates to the plain network baseline (no playouts run at all).
    func testRolloutCountZeroDegeneratesToNetworkValue() async throws {
        let ev = PassOnlyEvaluator(sizes: [9], expected: 0.7)
        let g = try GameState(boardSize: 9, komi: 7)
        var st = SearchSettings(); st.valueSource = .rollout(count: 0, maxMoves: 20, weightNetwork: 0.3)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 1)
        XCTAssertEqual(r.searchExpected, r.rootRawExpected, accuracy: 1e-9)   // == e_nn(black), unblended
        let diag = await search.rolloutDiagnostics()
        XCTAssertEqual(diag.leaves, 0)
        XCTAssertEqual(diag.playouts, 0)
    }

    /// docs/implementation-status.md 2026-09-10 §4-5: "search under `.rollout` with a fake
    /// evaluator whose wdl is uninformative still finds the winning move in a simple capture-race
    /// position (design the position so that random playouts strongly favour one move)". Position
    /// (9x9, komi 0, black to move): a 4-stone white group at (1,1),(2,1),(1,2),(2,2) is in atari,
    /// its sole liberty at (3,1) (7 of its other 8 perimeter points are already black); a fixed
    /// 3x3 white block elsewhere gives white a background lead. `winDrawLoss` is always 0.5
    /// (`.network` mode could never tell these moves apart). With `RootUniformThenPassEvaluator`,
    /// every root move's rollout deterministically scores the board exactly as it stands right
    /// after that one move (docs/spec/03-engine.md §3-4): capturing at (3,1) removes the white
    /// group and converts its 4 points to black territory (white 9, black 7+1+4=12 -> black wins
    /// by 3); every other legal move leaves the white group alive (white 9+4=13, black 7+1=8 ->
    /// white wins by 5). Only the capturing move crosses from a loss to a win for black.
    func testSearchUnderRolloutFindsWinningMoveInCaptureRace() async throws {
        var stones: [(player: Player, x: Int, y: Int)] = []
        // 7 of the atari group's 8 perimeter points (the 8th, (3,1), is its last liberty).
        for p in [(1, 0), (2, 0), (1, 3), (2, 3), (0, 1), (0, 2), (3, 2)] { stones.append((.black, p.0, p.1)) }
        // The 4-stone white group in atari.
        for p in [(1, 1), (2, 1), (1, 2), (2, 2)] { stones.append((.white, p.0, p.1)) }
        // A fixed, uninvolved 3x3 white block far from the capture race (background lead).
        for x in 5 ... 7 { for y in 5 ... 7 { stones.append((.white, x, y)) } }

        let ev = RootUniformThenPassEvaluator(sizes: [9])
        let g = try GameState(boardSize: 9, komi: 0, initialStones: stones, initialPlayer: .black)
        var st = SearchSettings(); st.leafBatch = 8
        st.valueSource = .rollout(count: 1, maxMoves: 4, weightNetwork: 0)   // pure rollout value; count 1 suffices (fully deterministic)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 200)
        XCTAssertEqual(r.move, .point(x: 3, y: 1))
    }

    func testNodeBudgetStopsExpansion() async throws {
        let ev = FakeEvaluator(sizes: [9], expected: 0.5)
        var st = SearchSettings(); st.maxNodes = 10
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: st, initial: g.record)
        let r = try await search.run(visits: 100)
        let n = await search.nodeCountForTests()
        XCTAssertLessThanOrEqual(n, 10)
        XCTAssertLessThan(r.rootVisits, 100)
        XCTAssertTrue(g.isLegal(.black, r.move))
    }
}
