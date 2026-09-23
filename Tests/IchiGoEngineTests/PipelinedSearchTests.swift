import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import XCTest

/// `SearchSettings.pipelinedEvaluation` (`Search.runPipelined`): one batch in the evaluator while
/// the next is built. Same contract as the sequential loop — exact visit target, legal moves, no
/// reservation ever leaked (failure, stale generation), deadline stops new batches.
final class PipelinedSearchTests: XCTestCase {
    private func pipelined(batch: Int = 8) -> SearchSettings {
        var s = SearchSettings()
        s.pipelinedEvaluation = true
        s.leafBatch = batch
        return s
    }

    func testCompletesExactlyWithLegalMovesAndNoLeakedReservations() async throws {
        for S in [9, 19] {
            let ev = FakeEvaluator(sizes: [S], expected: 0.5, favoured: [S * S / 2])
            let g = try GameState(boardSize: S, komi: 7)
            try g.play(.black, .point(x: 2, y: 2))
            let search = try Search(evaluator: ev, modelHash: "fake", settings: pipelined(), initial: g.record)
            let r = try await search.run(visits: 200)
            XCTAssertEqual(r.rootVisits, 200)
            XCTAssertEqual(r.candidates.reduce(0) { $0 + $1.visits }, r.rootVisits - 1)
            XCTAssertTrue(g.isLegal(.white, r.move))
            let sizes = await ev.batchSizes
            XCTAssertTrue(sizes.dropFirst().allSatisfy { $0 <= 8 })
            XCTAssertGreaterThan(sizes.count, 10)
            let reserved = await search.pendingReservationsForTests()
            XCTAssertEqual(reserved, 0)
        }
    }

    func testFailureReleasesEveryBatchAndSearchContinues() async throws {
        let ev = FakeEvaluator(sizes: [9], expected: 0.5)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: pipelined(), initial: g.record)
        await ev.setFailAfter(3)
        do { _ = try await search.run(visits: 100); XCTFail("expected failure") } catch {}
        var reserved = await search.pendingReservationsForTests()
        XCTAssertEqual(reserved, 0)
        await ev.setFailAfter(nil)
        let r = try await search.run(visits: 100)
        XCTAssertEqual(r.rootVisits, 100)
        XCTAssertEqual(r.candidates.reduce(0) { $0 + $1.visits }, r.rootVisits - 1)
        reserved = await search.pendingReservationsForTests()
        XCTAssertEqual(reserved, 0)
    }

    func testStopsBeforeDeadline() async throws {
        let clock = SystemMonotonicClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 0.03)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: pipelined(), initial: g.record, clock: clock)
        let start = await clock.now()
        let r = try await search.run(visits: 100_000, deadline: start + 0.08)
        let elapsed = await clock.now() - start
        XCTAssertGreaterThan(r.rootVisits, 0)
        XCTAssertLessThan(r.rootVisits, 100_000)
        XCTAssertLessThan(elapsed, 1.0)
        XCTAssertTrue(g.isLegal(.black, r.move))
        let reserved = await search.pendingReservationsForTests()
        XCTAssertEqual(reserved, 0)
    }

    func testBatchInFlightDuringMakeMoveIsDroppedAndReleased() async throws {
        let clock = FakeClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 10)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", settings: pipelined(), initial: g.record, clock: clock)
        let stale = Task { try await search.run(visits: 50) }
        await clock.waitForWaiters(1)   // root evaluation parked
        await clock.set(10)             // root returns; the first pipelined batch parks until 20
        await clock.waitForWaiters(1)

        try await search.makeMove(.point(x: 4, y: 4))
        let genAfterCommit = await search.currentGeneration()
        let visitsAfterCommit = await search.rootVisits()

        await clock.set(20)             // the stale batch finally returns
        do {
            _ = try await stale.value
            XCTFail("the orphaned run() should report its generation as invalidated")
        } catch {}
        let gen = await search.currentGeneration()
        let visits = await search.rootVisits()
        let reserved = await search.pendingReservationsForTests()
        XCTAssertEqual(gen, genAfterCommit)
        XCTAssertEqual(visits, visitsAfterCommit)
        XCTAssertEqual(reserved, 0)
    }
}
