import IchiGoFeatures

/// Wraps one `Search.run` in a wall-clock watchdog (docs/spec/03-engine.md §8): "GPUの処理中断ができ
/// なくても、deadline watchdogで応答経路を解放し、遅延結果を次手に混入させない" — even when the
/// evaluator cannot be interrupted mid-flight, the caller must get an answer by `deadline`, and
/// whatever the abandoned search eventually produces must never be committed.
///
/// ## Design
/// The search runs as an **unstructured** `Task` (not a `TaskGroup` child): Swift's structured
/// concurrency implicitly awaits every `TaskGroup` child before the group scope returns, which
/// would defeat the watchdog entirely for a non-cancellable evaluator (the whole point is to
/// *not* wait for it). A plain `Task` has no such obligation — if the watchdog wins, this function
/// returns immediately and the search task is left to finish (or not) in the background.
///
/// A `CommitGate` actor arbitrates which of the two racers — the search finishing on its own, or
/// the watchdog's timer elapsing — gets to commit. `tryCommit()` returns `true` to at most one
/// caller ever, so **exactly one** of "play the search's own move" and "play the fallback move"
/// happens, no matter how close the race is (including a search that returns at the exact
/// deadline instant). The loser never calls `Search.makeMove`, so there is never a second commit
/// to race against — this is what the tests in `TimeTests.swift` exercise directly.
///
/// The winning branch always calls `Search.makeMove` itself (this is the one place that commits a
/// move into the search tree), which — win or lose — bumps `Search`'s internal generation. That
/// means a stale result the abandoned search task eventually produces is dropped by `Search`'s own
/// generation guard (see `Search.run`/`evaluateBatch`) the moment it tries to use it; no additional
/// bookkeeping is needed here to "invalidate" the loser.
public enum DeadlineController {
    public struct Outcome: Sendable, Equatable {
        public let move: MoveCoord
        /// The search's own result, when the search itself won the race (`timedOut == false`).
        public let result: SearchResult?
        public let timedOut: Bool
    }

    private actor CommitGate {
        private var committed = false
        /// `true` for the first caller only; every later caller gets `false`.
        func tryCommit() -> Bool {
            guard !committed else { return false }
            committed = true
            return true
        }
    }

    /// Runs `search.run(visits:deadline:)`. With `deadline == nil` this simply awaits it and
    /// commits its move — no watchdog, matching fixed-visits genmove (no `time_settings`).
    /// With a deadline, races it against `clock.sleep(until: deadline)`; whichever finishes first
    /// commits exactly one move (`Search.makeMove`) and this call returns as soon as that happens,
    /// without waiting for the loser.
    ///
    /// Fallback order when the watchdog wins (docs/spec/03-engine.md §8): the best root child by
    /// visits if the root has been evaluated and visited (`Search.savedRootCandidate`); else the
    /// legal move with the highest policy prior if a root evaluation exists at all
    /// (`Search.rootPolicyBestMove`); else the smallest-index legal point
    /// (`Search.legalMovesAscendingFallback`); else pass. All three reads are synchronous w.r.t.
    /// the NN (no evaluator call), so the fallback never itself waits on a hung evaluator.
    public static func run(search: Search, visits: Int, deadline: Double?, clock: any MonotonicClock) async throws -> Outcome {
        guard let deadline else {
            let r = try await search.run(visits: visits, deadline: nil)
            try await search.makeMove(r.move)
            return Outcome(move: r.move, result: r, timedOut: false)
        }

        let gate = CommitGate()
        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Outcome, Error>) in
            let searchTask = Task<Void, Never> {
                do {
                    let r = try await search.run(visits: visits, deadline: deadline)
                    guard await gate.tryCommit() else { return }   // watchdog already committed
                    do {
                        try await search.makeMove(r.move)
                        cont.resume(returning: Outcome(move: r.move, result: r, timedOut: false))
                    } catch {
                        cont.resume(throwing: error)
                    }
                } catch {
                    // A genuine search failure (or `run` throwing after losing an earlier race —
                    // see `Search`'s generation guard). Only surface it if nobody has answered yet;
                    // otherwise this is exactly the late result the watchdog path already discarded.
                    if await gate.tryCommit() { cont.resume(throwing: error) }
                }
            }
            Task<Void, Never> {
                try? await clock.sleep(until: deadline)
                guard await gate.tryCommit() else { return }   // the search already committed its own move
                do {
                    let move = await fallbackMove(search: search)
                    try await search.makeMove(move)
                    cont.resume(returning: Outcome(move: move, result: nil, timedOut: true))
                } catch {
                    cont.resume(throwing: error)
                }
                searchTask.cancel()   // best-effort only: a non-cooperative (e.g. GPU) evaluator may ignore this
            }
        }
    }

    static func fallbackMove(search: Search) async -> MoveCoord {
        if let saved = await search.savedRootCandidate() { return saved }
        if let byPolicy = await search.rootPolicyBestMove() { return byPolicy }
        return await search.legalMovesAscendingFallback().first ?? .pass
    }
}
