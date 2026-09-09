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
