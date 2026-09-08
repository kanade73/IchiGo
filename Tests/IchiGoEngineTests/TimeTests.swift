import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import LogicModel
import XCTest

/// Deterministic clock for tests (docs/spec/03-engine.md §8): `now()` only changes when a test
/// calls `advance`/`set`, and `sleep(until:)` suspends until some caller moves the clock to (or
/// past) that instant. That is what makes an "exact tie at the deadline" reproducible instead of
/// depending on real scheduler timing — a `DelayedEvaluator` below parks on this clock exactly
/// like `DeadlineController`'s watchdog does, so a test can force both racers to become ready at
/// the same simulated instant.
actor FakeClock: MonotonicClock {
    private var t: Double
    private var waiters: [(threshold: Double, cont: CheckedContinuation<Void, Never>)] = []

    init(start: Double = 0) { t = start }

    func now() -> Double { t }

    func sleep(until instant: Double) async {
        if t >= instant { return }
        await withCheckedContinuation { cont in waiters.append((instant, cont)) }
    }

    /// Moves the clock forward to `value` (never backwards) and resumes every waiter whose
    /// threshold has now been reached.
    func set(_ value: Double) {
        guard value > t else { return }
        t = value
        let ready = waiters.filter { $0.threshold <= t }
        waiters.removeAll { $0.threshold <= t }
        for w in ready { w.cont.resume() }
    }

    func advance(by dt: Double) { set(t + dt) }

    /// Test-only synchronization: waits (via cooperative yields — no simulated time passes) until
    /// at least `n` callers are parked in `sleep(until:)`, so a test can be sure every racer has
    /// registered before it advances the clock past their thresholds.
    func waitForWaiters(_ n: Int) async {
        while waiters.count < n { await Task.yield() }
    }
}

/// Fake evaluator (Tests only) whose response can be delayed by parking on the injected clock —
/// works with both `FakeClock` (fully deterministic, no real waiting) and `SystemMonotonicClock`
/// (a real, bounded delay for the "does the loop actually stop early" sanity tests below).
actor DelayedEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    private let clock: any MonotonicClock
    private let expected: Float
    private let favoured: [Int]
    var delay: Double
    private(set) var calls = 0

    init(sizes: Set<Int>, clock: any MonotonicClock, expected: Float = 0.5, favoured: [Int] = [], delay: Double = 0) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
        self.clock = clock
        self.expected = expected
        self.favoured = favoured
        self.delay = delay
    }

    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        calls += 1
        if delay > 0 {
            let now = await clock.now()
            try await clock.sleep(until: now + delay)
        }
        return positions.map { s in
            let P = s.boardSize * s.boardSize
            var policy = [Float](repeating: 0, count: P + 1)
            var mass: Float = 0
            for i in 0 ... P where s.legal[i] == 1 { policy[i] = favoured.contains(i) ? 10 : 1; mass += policy[i] }
            policy = policy.map { $0 / mass }
            return LogicEvaluation(policy: policy, winDrawLoss: [expected, 0, 1 - expected], expectedResult: expected, scoreMean: 0, ownership: [Float](repeating: 0, count: P))
        }
    }

    func preWarm(size: Int) async throws {}
}

final class TimeTests: XCTestCase {
    // MARK: - TimeManager (pure math, docs/spec/03-engine.md §8)

    func testBudgetSuddenDeathCases() {
        // R=0: reserve floors at 0.1 > R, so the (R-reserve) cap is negative → budget clamps to 0.
        XCTAssertEqual(TimeManager.budget(remaining: 0, moveNumber: 0, boardSize: 9), 0, accuracy: 1e-12)
        // R=0.05s: same story — nowhere near the 0.1s floor reserve.
        XCTAssertEqual(TimeManager.budget(remaining: 0.05, moveNumber: 0, boardSize: 9), 0, accuracy: 1e-12)
        // R=1s, 9x9, move 0: reserve=0.1, estimatedMoves=max(10,48.6)=48.6, budget=0.9/48.6.
        XCTAssertEqual(TimeManager.budget(remaining: 1, moveNumber: 0, boardSize: 9), 0.9 / 48.6, accuracy: 1e-9)
        // R=300s, 9x9, move 0: reserve clamps to 2.0, budget=298/48.6.
        XCTAssertEqual(TimeManager.budget(remaining: 300, moveNumber: 0, boardSize: 9), 298.0 / 48.6, accuracy: 1e-9)
        // A budget never exceeds what is left after the reserve, for any of the four R's, either size.
        for r in [0.0, 0.05, 1, 300] {
            for s in [9, 19] {
                let b = TimeManager.budget(remaining: r, moveNumber: 0, boardSize: s)
                XCTAssertGreaterThanOrEqual(b, 0)
                XCTAssertLessThanOrEqual(b, max(0, r - TimeManager.reserve(remaining: r)))
            }
        }
    }

    func testReserveClampedBetween0_1And2_0() {
        XCTAssertEqual(TimeManager.reserve(remaining: 0), 0.1, accuracy: 1e-12)
        XCTAssertEqual(TimeManager.reserve(remaining: 5), 0.1, accuracy: 1e-12)     // 0.01*5=0.05 < floor
        XCTAssertEqual(TimeManager.reserve(remaining: 50), 0.5, accuracy: 1e-12)    // 0.01*50=0.5, between floor/cap
        XCTAssertEqual(TimeManager.reserve(remaining: 1000), 2.0, accuracy: 1e-12)  // 0.01*1000=10 > cap
    }

    func testEstimatedMovesFloorsAt10() {
        XCTAssertEqual(TimeManager.estimatedMoves(boardSize: 9, moveNumber: 0), 48.6, accuracy: 1e-9)
        XCTAssertEqual(TimeManager.estimatedMoves(boardSize: 9, moveNumber: 200), 10)      // would go negative otherwise
        XCTAssertEqual(TimeManager.estimatedMoves(boardSize: 19, moveNumber: 0), 216.6, accuracy: 1e-9)
    }

    func testStopMarginFloorAndP95() {
        XCTAssertEqual(TimeManager.stopMargin(recentBatchDurations: []), 0.01, accuracy: 1e-12)
        XCTAssertEqual(TimeManager.stopMargin(recentBatchDurations: [0.001]), 0.01, accuracy: 1e-12)  // floor dominates
        let xs = Array(repeating: 0.02, count: 20)
        XCTAssertEqual(TimeManager.stopMargin(recentBatchDurations: xs), 0.04, accuracy: 1e-9)         // 2*p95 dominates
    }

    func testValidateSuddenDeathAcceptsOnlyByo0Stones0() {
        XCTAssertNoThrow(try TimeManager.validateSuddenDeath(byo: 0, stones: 0))
        XCTAssertThrowsError(try TimeManager.validateSuddenDeath(byo: 30, stones: 5))
        XCTAssertThrowsError(try TimeManager.validateSuddenDeath(byo: 30, stones: 0))
        XCTAssertThrowsError(try TimeManager.validateSuddenDeath(byo: 0, stones: 1))
    }

    // MARK: - Search fallback tiers (DeadlineController's non-NN answers)

    func testFallbackTiersDirectly() async throws {
        let clock = FakeClock()
        let g = try GameState(boardSize: 9, komi: 7)

        // Tier 3: nothing evaluated at all → smallest-index legal point.
        let s1 = try Search(evaluator: DelayedEvaluator(sizes: [9], clock: clock), modelHash: "fake", initial: g.record, clock: clock)
        let saved1 = await s1.savedRootCandidate()
        let policy1 = await s1.rootPolicyBestMove()
        XCTAssertNil(saved1)
        XCTAssertNil(policy1)
        let legal1 = await s1.legalMovesAscendingFallback()
        XCTAssertEqual(legal1.first, .point(x: 0, y: 0))

        // Tier 2: root evaluated (policy priors exist) but no child has been visited yet.
        let ev2 = DelayedEvaluator(sizes: [9], clock: clock, favoured: [40])
        let s2 = try Search(evaluator: ev2, modelHash: "fake", initial: g.record, clock: clock)
        _ = try await s2.run(visits: 1)   // only the root's own (mandatory) evaluation
        let saved2 = await s2.savedRootCandidate()
        XCTAssertNil(saved2)
        let policy2 = await s2.rootPolicyBestMove()
        XCTAssertEqual(policy2, .point(x: 4, y: 4))   // index 40 on 9x9, the favoured (highest-prior) move

        // Tier 1: at least one root child has actually been visited.
        var st = SearchSettings(); st.leafBatch = 1
        let ev3 = DelayedEvaluator(sizes: [9], clock: clock, favoured: [40])
        let s3 = try Search(evaluator: ev3, modelHash: "fake", settings: st, initial: g.record, clock: clock)
        _ = try await s3.run(visits: 3)
        let saved3 = await s3.savedRootCandidate()
        XCTAssertEqual(saved3, .point(x: 4, y: 4))
    }

    // MARK: - Search(visits:deadline:) stop margin (real clock, bounded real delay)

    /// A genuinely slow (real Task.sleep-backed) evaluator paired with a short real deadline: the
    /// loop must stop well short of the (huge) visits target, close to the deadline, not run away.
    func testRunStopsBeforeDeadlineInsteadOfReachingVisitsTarget() async throws {
        let clock = SystemMonotonicClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 0.03)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)
        let start = await clock.now()
        let deadline = start + 0.08
        let r = try await search.run(visits: 100_000, deadline: deadline)
        let elapsed = await clock.now() - start
        XCTAssertGreaterThan(r.rootVisits, 0)
        XCTAssertLessThan(r.rootVisits, 100_000)
        XCTAssertLessThan(elapsed, 1.0)   // generous: should be a couple of 30ms batches, not a runaway loop
        XCTAssertTrue(g.isLegal(.black, r.move))
    }

    // MARK: - DeadlineController

    func testNormalCompletionCommitsTheSearchsOwnMove() async throws {
        let clock = FakeClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, expected: 0.6, favoured: [40])
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)
        // delay=0 → the evaluator never parks on the clock, so `run` finishes purely by reaching
        // `visits`; the deadline (frozen far in the future on this fake clock) never binds.
        let outcome = try await DeadlineController.run(search: search, visits: 5, deadline: 1000, clock: clock)
        XCTAssertFalse(outcome.timedOut)
        XCTAssertEqual(outcome.result?.rootVisits, 5)
        XCTAssertTrue(g.isLegal(.black, outcome.move))
        let genAfter = await search.currentGeneration()
        XCTAssertEqual(genAfter, 1)   // exactly one commit
        await clock.set(1000)   // release the now-pointless watchdog wait (continuation hygiene)
    }

    func testZeroBudgetDeadlineAlreadyElapsedStillReturnsLegalMove() async throws {
        let clock = FakeClock(start: 5)
        let ev = DelayedEvaluator(sizes: [9], clock: clock)   // delay 0: effectively instant
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)
        let outcome = try await DeadlineController.run(search: search, visits: 100, deadline: 5, clock: clock)   // deadline == now
        XCTAssertTrue(g.isLegal(.black, outcome.move))
        let genAfter = await search.currentGeneration()
        XCTAssertEqual(genAfter, 1)   // exactly one commit either way
    }

    /// Evaluator sleeps far longer than the deadline: the watchdog must win, return a legal
    /// fallback move promptly, and commit it exactly once.
    func testSlowEvaluatorFallsBackWithinDeadlineAndCommitsExactlyOnce() async throws {
        let clock = FakeClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 1000)   // "never" relative to the deadline
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)
        async let outcome = DeadlineController.run(search: search, visits: 50, deadline: 1, clock: clock)
        await clock.waitForWaiters(2)   // the root eval (parked at t=1000) and the watchdog (parked at t=1)
        await clock.set(1)              // only cross the deadline — the evaluator stays parked
        let o = try await outcome
        XCTAssertTrue(o.timedOut)
        XCTAssertNil(o.result)
        XCTAssertTrue(g.isLegal(.black, o.move))
        let genAfter = await search.currentGeneration()
        let movesAfter = await search.rootRecord().moves.count
        XCTAssertEqual(genAfter, 1)              // exactly one commit
        XCTAssertEqual(movesAfter, 1)
        await clock.set(1000)   // release the abandoned evaluator call (continuation hygiene only;
        // its generation-drop behaviour is covered directly by testLateResultAfterMakeMoveIsDropped below)
    }

    /// Forces the search to finish (its evaluator resolves) at the *exact* instant the watchdog's
    /// deadline elapses, repeatedly: `CommitGate` must still pick exactly one winner every time,
    /// regardless of which of the two unstructured tasks the scheduler happens to run first.
    func testExactDeadlineTieAlwaysCommitsExactlyOnce() async throws {
        for _ in 0 ..< 25 {
            let clock = FakeClock()
            let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 5)   // resolves at t=5, same as the deadline
            let g = try GameState(boardSize: 9, komi: 7)
            let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)
            async let outcome = DeadlineController.run(search: search, visits: 50, deadline: 5, clock: clock)
            await clock.waitForWaiters(2)
            await clock.set(5)   // release both racers at once
            let o = try await outcome
            XCTAssertTrue(g.isLegal(.black, o.move))
            let genAfter = await search.currentGeneration()
            let movesAfter = await search.rootRecord().moves.count
            XCTAssertEqual(genAfter, 1, "exactly one commit expected")
            XCTAssertEqual(movesAfter, 1)
        }
    }

    /// The generation-id drop, tested directly per docs/spec/03-engine.md §8: start a search that
    /// is blocked on the evaluator, invalidate it via `makeMove` (as `DeadlineController`'s
    /// watchdog path effectively does), then let the stale evaluator result arrive. The tree
    /// `makeMove` committed to must be completely unaffected by it, and the orphaned `run` call
    /// must report the invalidation rather than silently succeeding.
    func testLateResultAfterMakeMoveIsDroppedNotBackedUp() async throws {
        let clock = FakeClock()
        let ev = DelayedEvaluator(sizes: [9], clock: clock, delay: 10)
        let g = try GameState(boardSize: 9, komi: 7)
        let search = try Search(evaluator: ev, modelHash: "fake", initial: g.record, clock: clock)

        let staleRun = Task { try await search.run(visits: 5) }
        await clock.waitForWaiters(1)   // the root eval is parked on the evaluator

        try await search.makeMove(.point(x: 4, y: 4))
        let genAfterCommit = await search.currentGeneration()
        let visitsAfterCommit = await search.rootVisits()
        let movesAfterCommit = await search.rootRecord().moves.count
        XCTAssertEqual(movesAfterCommit, 1)

        await clock.set(10)   // let the stale evaluate() call finally return
        do {
            _ = try await staleRun.value
            XCTFail("the orphaned run() call should report its generation as invalidated")
        } catch {
            // expected — see Search.run's documentation of this behaviour
        }

        // Nothing about the committed tree changed because of the late result.
        let genAfterLateResult = await search.currentGeneration()
        let visitsAfterLateResult = await search.rootVisits()
        let movesAfterLateResult = await search.rootRecord().moves.count
        XCTAssertEqual(genAfterLateResult, genAfterCommit)
        XCTAssertEqual(visitsAfterLateResult, visitsAfterCommit)
        XCTAssertEqual(movesAfterLateResult, movesAfterCommit)
    }
}
