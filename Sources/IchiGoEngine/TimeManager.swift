import Foundation

/// Monotonic clock abstraction (docs/spec/03-engine.md §8) so time-budget code — and its tests —
/// never depend on the wall clock directly. `now()` returns seconds since an unspecified fixed
/// epoch; only differences between two calls on the *same* clock instance are meaningful.
/// `sleep(until:)` suspends the caller until the clock reaches (or has already passed) `instant`.
/// A fake clock (Tests only, see `Tests/IchiGoEngineTests/TimeTests.swift`) can resume waiters
/// itself instead of waiting on real time, which is what makes exact-deadline races testable.
public protocol MonotonicClock: Sendable {
    func now() async -> Double
    func sleep(until instant: Double) async throws
}

/// Real clock, backed by `DispatchTime`'s uptime counter (immune to wall-clock/NTP adjustments).
public struct SystemMonotonicClock: MonotonicClock {
    public init() {}

    public func now() async -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }

    public func sleep(until instant: Double) async throws {
        let dt = instant - (await now())
        guard dt > 0 else { return }
        try await Task.sleep(nanoseconds: UInt64((dt * 1_000_000_000).rounded(.up)))
    }
}

/// Per-move clock math (docs/spec/03-engine.md §8). Every function here is pure (no I/O, no
/// clock reads) — callers capture `clock.now()` themselves and pass plain `Double` seconds, which
/// keeps this type trivially testable and keeps the monotonic-clock dependency isolated to the
/// call sites (`GTPEngine`, `Search`, `DeadlineController`) that actually need to read the time.
public enum TimeManager {
    /// v1 only supports GTP sudden-death time control (`byo=0`, `stones=0`); any real byo-yomi
    /// period is an explicit, permanent-for-v1 rejection at `time_settings` (docs/spec/03-engine.md
    /// §8 — "byo>0はv1でエラーとして未対応を明示").
    public static func validateSuddenDeath(byo: Double, stones: Int) throws {
        guard byo == 0, stones == 0 else {
            throw SearchError(message: "byo-yomi is not supported in v1 (sudden death only)")
        }
    }

    /// Reserve held back from `remaining` seconds so a move is never budgeted the *entire* clock:
    /// `max(0.1, min(2.0, 0.01*R))`.
    public static func reserve(remaining: Double) -> Double {
        max(0.1, min(2.0, 0.01 * remaining))
    }

    /// Rough remaining-move estimate used to spread `remaining` over the rest of the game:
    /// `max(10, 0.6*S*S - m/2)`, `S` = board size, `m` = move number already played.
    public static func estimatedMoves(boardSize: Int, moveNumber: Int) -> Double {
        max(10, 0.6 * Double(boardSize * boardSize) - Double(moveNumber) / 2)
    }

    /// Per-move time budget in seconds, sudden death only (docs/spec/03-engine.md §8):
    /// `budget = max(0, min((R-reserve)/estimatedMoves, R-reserve))`.
    /// This budget is meant to cover the *entire* move, including feature encoding and the first
    /// (root) NN evaluation — callers must start their deadline clock before any of that work, not
    /// just before the search loop's leaf batches.
    public static func budget(remaining: Double, moveNumber: Int, boardSize: Int) -> Double {
        let r = reserve(remaining: remaining)
        let cap = remaining - r
        let moves = estimatedMoves(boardSize: boardSize, moveNumber: moveNumber)
        return max(0, min(cap / moves, cap))
    }

    /// The 95th percentile of `values` (linear interpolation between the two nearest ranks, same
    /// convention as `numpy.percentile`'s default). `0` for an empty input.
    public static func percentile95(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        guard sorted.count > 1 else { return sorted[0] }
        let rank = 0.95 * Double(sorted.count - 1)
        let lo = Int(rank.rounded(.down))
        let hi = Int(rank.rounded(.up))
        let frac = rank - Double(lo)
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }

    /// Margin kept before the deadline for issuing one more leaf batch (docs/spec/03-engine.md
    /// §8): `max(0.01, 2 * p95(recentBatchDurations))`. With no batches timed yet this is just the
    /// floor, `0.01`s.
    public static func stopMargin(recentBatchDurations: [Double]) -> Double {
        max(0.01, 2 * percentile95(recentBatchDurations))
    }
}
