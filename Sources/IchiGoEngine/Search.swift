import Foundation
import IchiGoCore
import IchiGoFeatures
import LogicModel

/// Pure-tree PUCT search v1 (docs/spec/03-engine.md §4). Structure follows RinGo's `Search` actor
/// (nodes owned by the actor, batched leaf collection with in-flight reservations, tree reuse on
/// `makeMove`), but the numerics are exactly the spec's: no graph merge, no LCB, no uncertainty,
/// score utility factor 0, win/loss utility factor 1, FPU reduction 0.
///
///   Qparent(a) = parent is white ? Qwhite(a) : -Qwhite(a)
///   U(a)       = cpuct * prior(a) * sqrt(max(1, Nparent)) / (1 + Nedge(a) + reservations(a))
///   select     = argmax(Qparent + U); ties → smaller policy index
/// Unvisited Q = parent's NN white value (as seen by the parent). Backups are always white
/// perspective. The root's first NN evaluation counts as one visit. Terminal leaves use the exact
/// Core outcome and never call the network.
public struct SearchSettings: Sendable {
    public var cpuct: Double = 1.5
    public var fpuReduction: Double = 0
    public var maxNodes: Int = 100_000
    public var leafBatch: Int = 8
    public var treeReuse: Bool = true
    public var virtualLossValue: Double = -1   // white value assumed for a reserved (in-flight) edge, in the parent's view
    /// Leaf value source (docs/spec/03-engine.md §3-4 "値ソース"). `.network` (default) keeps the
    /// current behaviour (the NN's own wdl head, unmodified). `.ownership`/`.blend` route leaf
    /// expansion through `EvaluationAdapter.toWhite`'s ownership-derived value instead of, or
    /// blended with, the network's. `.rollout` runs policy-guided playouts through `Search`
    /// itself (`applyRollout`), since that needs the async evaluator and `rolloutRNGSeed` below.
    /// `SearchResult.rootRawExpected`/`rootRawWinDrawLoss` always reflect the raw network output
    /// regardless of this setting.
    public var valueSource: ValueSource = .network
    /// Seed for the `SplitMix64` RNG that samples playout moves for `ValueSource.rollout`
    /// (docs/spec/03-engine.md §3-4 "値ソース"). Carried in settings — rather than injected
    /// separately — so a whole search's playout sequence is reproducible from its
    /// `SearchSettings` alone, given a deterministic evaluator. Unused by any other value source.
    public var rolloutRNGSeed: UInt64 = 0x726F_6C6C_6F75_74

    public init() {}
}

public struct SearchError: Error, CustomStringConvertible, Equatable {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

public struct MoveCandidate: Sendable, Equatable {
    public let move: MoveCoord
    public let index: Int
    public let visits: Int
    public let prior: Float
    /// Expected score for the root's side to move in [0,1] (draw = 0.5).
    public let expectedScore: Double
    /// Score lead for the root's side to move.
    public let scoreLead: Double
    public let order: Int
    public let pv: [MoveCoord]
}

/// Per-`run()`-call playout accounting for `ValueSource.rollout` (docs/implementation-status.md
/// 2026-09-10 §4-5: "log per-genmove the average playouts per leaf and the fraction of playouts
/// that hit maxMoves"). Reset at the start of every `Search.run(visits:deadline:)` call; queried
/// via `Search.rolloutDiagnostics()` after the call returns. All zero when `valueSource` isn't
/// `.rollout`, or before any leaf has been expanded.
public struct RolloutDiagnostics: Sendable, Equatable {
    public let leaves: Int
    public let playouts: Int
    public let maxMovesHit: Int
    public var averagePlayoutsPerLeaf: Double { leaves > 0 ? Double(playouts) / Double(leaves) : 0 }
    public var maxMovesHitFraction: Double { playouts > 0 ? Double(maxMovesHit) / Double(playouts) : 0 }
}

public struct SearchResult: Sendable, Equatable {
    public let move: MoveCoord
    public let rootVisits: Int
    public let candidates: [MoveCandidate]
    /// Root NN raw expected score (side to move) and search expected score (side to move).
    public let rootRawExpected: Double
    public let searchExpected: Double
    public let searchScoreLead: Double
    public let rootRawWinDrawLoss: [Float]
    public let toMove: Player
    public let modelHash: String
}

final class SearchNode {
    let state: GameState
    let toMove: Player
    let terminal: (value: Float, score: Float)?   // white perspective exact outcome
    var evaluated = false
    var nnWhiteValue: Double = 0         // white expected*2-1 from the NN
    var nnWhiteScore: Double = 0
    var rawExpected: Double = 0.5
    var rawWDL: [Float] = [0, 0, 0]
    var visits: Int = 0
    var whiteValueSum: Double = 0
    var whiteScoreSum: Double = 0
    var edges: [Edge] = []
    var stateFingerprint: String?

    struct Edge {
        let index: Int
        let move: MoveCoord
        let prior: Double
        var child: SearchNode?
        var visits: Int = 0
        var reservations: Int = 0
    }

    init(state: GameState) {
        self.state = state
        toMove = state.toMove
        if let o = state.exactWhiteOutcome { terminal = (o.value, o.whiteMinusBlack) } else { terminal = nil }
    }

    var meanWhiteValue: Double { visits > 0 ? whiteValueSum / Double(visits) : nnWhiteValue }
    var meanWhiteScore: Double { visits > 0 ? whiteScoreSum / Double(visits) : nnWhiteScore }
}

/// A selected root-to-leaf path and the edge used to enter each child. Keeping the edge indices
/// alongside the nodes makes reservation release and backup O(path depth), rather than scanning
/// every parent's edge array to rediscover the child.
private struct SearchPath {
    var nodes: [SearchNode]
    var edgeIndices: [Int]
}

public actor Search {
    public let settings: SearchSettings
    private let evaluator: any PositionEvaluating
    private let modelHash: String
    private let clock: any MonotonicClock
    private var root: SearchNode
    private var nodeCount = 1
    private var generation = 0
    /// Wall-clock duration of the most recent leaf batches (bounded window), used to compute the
    /// deadline stop margin (docs/spec/03-engine.md §8, `TimeManager.stopMargin`). Only populated
    /// by time-limited `run(visits:deadline:)` calls.
    private var recentBatchDurations: [Double] = []
    /// RNG for `ValueSource.rollout` playout move sampling, seeded from `settings.rolloutRNGSeed`.
    /// Advances across the whole search's lifetime (never reset per leaf or per `run()` call) so a
    /// full genmove's playout sequence is reproducible from the settings alone.
    private var rolloutRNG: SplitMix64
    /// `ValueSource.rollout` diagnostics accumulated by the current `run()` call; reset at its
    /// start. See `RolloutDiagnostics`/`rolloutDiagnostics()`.
    private var rolloutLeaves = 0
    private var rolloutPlayouts = 0
    private var rolloutPlayoutsHitMaxMoves = 0

    public init(evaluator: any PositionEvaluating, modelHash: String, settings: SearchSettings = SearchSettings(), initial: GameRecord, clock: any MonotonicClock = SystemMonotonicClock()) throws {
        self.evaluator = evaluator
        self.modelHash = modelHash
        self.settings = settings
        self.clock = clock
        rolloutRNG = SplitMix64(seed: settings.rolloutRNGSeed)
        root = SearchNode(state: try GameState(record: initial))
    }

    // MARK: - state management

    /// Replaces the root with a fresh copy of `state`, discarding the tree.
    public func reset(to record: GameRecord) throws {
        root = SearchNode(state: try GameState(record: record))
        nodeCount = 1
        generation += 1
    }

    public func rootRecord() -> GameRecord { root.state.record }
    public func rootVisits() -> Int { root.visits }
    public func nodeCountForTests() -> Int { nodeCount }
    public func currentGeneration() -> Int { generation }
    /// `ValueSource.rollout` playout accounting for the most recent `run(visits:deadline:)` call
    /// (reset at its start). All zero when `valueSource` isn't `.rollout`.
    public func rolloutDiagnostics() -> RolloutDiagnostics {
        RolloutDiagnostics(leaves: rolloutLeaves, playouts: rolloutPlayouts, maxMovesHit: rolloutPlayoutsHitMaxMoves)
    }

    /// Plays `move` at the root. Promotes the existing child subtree when tree reuse is enabled
    /// and the child's full-state fingerprint equals the freshly replayed position; otherwise the
    /// tree is discarded. Throws on an illegal move.
    public func makeMove(_ move: MoveCoord) throws {
        let next = root.state.copy()
        guard next.isLegal(next.toMove, move) else { throw SearchError(message: "illegal move \(move)") }
        try next.play(next.toMove, move, assumeLegal: true)
        generation += 1
        if settings.treeReuse, let edge = root.edges.first(where: { $0.move == move }), let child = edge.child,
           (child.stateFingerprint ?? child.state.fingerprintValue()) == next.fingerprintValue() {
            root = child
            nodeCount = countNodes(root)
            return
        }
        root = SearchNode(state: next)
        nodeCount = 1
    }

    private func countNodes(_ n: SearchNode) -> Int {
        1 + n.edges.compactMap(\.child).reduce(0) { $0 + countNodes($1) }
    }

    // MARK: - search

    /// Runs until the root has `visits` visits (including the root's own evaluation), the node
    /// budget is exhausted, the task is cancelled, or (when `deadline` is given, as a monotonic
    /// instant from the injected `clock`) the clock is within the stop margin of the deadline.
    /// Returns the chosen move (max edge visits, ties → prior, then policy index) computed from
    /// whatever the tree looks like at that point — with a deadline this may be far short of
    /// `target`.
    ///
    /// The stop margin (`TimeManager.stopMargin`, `max(0.01, 2 * p95(recent batch durations))`) is
    /// checked *before* issuing each new leaf batch, so an in-flight batch is never abandoned —
    /// only the next one is skipped. The root's own (mandatory) first evaluation is never skipped
    /// by the deadline: a caller that cannot even wait for that should race this call with
    /// `DeadlineController` instead of expecting `run` to bail out early.
    ///
    /// Every request this method issues to `evaluator.evaluate` is tagged with the generation
    /// captured at the start of this call. `Search` is an actor, so the `await` on that call is a
    /// reentrancy point: another task can call `makeMove`/`reset` on the same actor while this one
    /// is suspended, which bumps `generation` and (on `makeMove`) replaces `root` — possibly with a
    /// node this very call's in-flight path passes through. If the generation has moved by the
    /// time the evaluator answers, the result is a *late result from a previous generation*: it is
    /// dropped without ever mutating the tree (no `expand`/`backup`), and only the reservations
    /// placed on that path are released, so a reused subtree is never left with permanent phantom
    /// virtual losses. This call itself still throws once it notices its generation is stale (there
    /// is no partial result worth returning under a generation that no longer owns `root`) — but
    /// nothing it observed after the mismatch was ever applied to the tree.
    public func run(visits target: Int, deadline: Double? = nil) async throws -> SearchResult {
        let gen = generation
        rolloutLeaves = 0
        rolloutPlayouts = 0
        rolloutPlayoutsHitMaxMoves = 0
        if root.terminal != nil { return try result(chosen: .pass) }
        if !root.evaluated {
            try await evaluateBatch([SearchPath(nodes: [root], edgeIndices: [])], generation: gen)
            guard gen == generation else { throw SearchError(message: "search invalidated") }
        }
        while root.visits < target, !Task.isCancelled {
            if let deadline {
                // `clock.now()` is itself a suspension point on this actor (any `await` is), so a
                // concurrent `makeMove`/`reset` could run before it returns — re-check generation
                // before trusting `root`/`settings` again.
                let now = await clock.now()
                guard gen == generation else { throw SearchError(message: "search invalidated") }
                if now + TimeManager.stopMargin(recentBatchDurations: recentBatchDurations) >= deadline { break }
            }
            let batch = collectBatch(limit: min(settings.leafBatch, target - root.visits))
            if batch.isEmpty { break }
            let batchStart = deadline != nil ? await clock.now() : nil
            guard gen == generation else { releaseAll(batch); throw SearchError(message: "search invalidated") }
            do {
                try await evaluateBatch(batch, generation: gen)
            } catch {
                releaseAll(batch)
                throw error
            }
            if let batchStart { recordBatchDuration(await clock.now() - batchStart) }
            guard gen == generation else { throw SearchError(message: "search invalidated") }
        }
        return try result(chosen: chooseMove())
    }

    private func recordBatchDuration(_ d: Double) {
        recentBatchDurations.append(d)
        if recentBatchDurations.count > 32 { recentBatchDurations.removeFirst(recentBatchDurations.count - 32) }
    }

    /// Best root child by visits (ties → prior, then index) — fallback tier 1 for
    /// `DeadlineController`. `nil` until at least one leaf batch beyond the root's own evaluation
    /// has completed.
    public func savedRootCandidate() -> MoveCoord? {
        var best: SearchNode.Edge? = nil
        for e in root.edges where e.visits > 0 {
            if best == nil || e.visits > best!.visits || (e.visits == best!.visits && e.prior > best!.prior) { best = e }
        }
        return best?.move
    }

    /// Root child with the highest policy prior (ties → index) — fallback tier 2. `nil` until the
    /// root has been NN-evaluated at all.
    public func rootPolicyBestMove() -> MoveCoord? {
        guard root.evaluated else { return nil }
        var best: SearchNode.Edge? = nil
        for e in root.edges where best == nil || e.prior > best!.prior { best = e }
        return best?.move
    }

    /// Legal moves at the root in ascending point index order, pass last — fallback tier 3
    /// (smallest-index legal point; the caller falls back to pass itself if this is empty). Reads
    /// `root.state` directly, so it never needs (or waits for) an NN evaluation.
    public func legalMovesAscendingFallback() -> [MoveCoord] {
        let snap = root.state.snapshot()
        var out: [MoveCoord] = []
        for i in snap.legal.indices where snap.legal[i] == 1 {
            out.append(try! Coordinates.move(fromIndex: i, size: snap.boardSize))
        }
        return out
    }

    private func collectBatch(limit: Int) -> [SearchPath] {
        var paths: [SearchPath] = []
        while paths.count < limit {
            guard let path = descend() else { break }
            paths.append(path)
            if path.nodes.last?.terminal != nil { continue }
        }
        return paths
    }

    /// Selects a path from the root; reserves every traversed edge. Expands one new node or
    /// stops at a terminal. Returns nil when the node budget is exhausted.
    private func descend() -> SearchPath? {
        var path = SearchPath(nodes: [root], edgeIndices: [])
        var node = root
        while true {
            if node.terminal != nil { return path }
            if !node.evaluated { return path }  // freshly expanded (or reserved) leaf awaiting evaluation
            guard let idx = selectEdge(node) else { return path }
            node.edges[idx].reservations += 1
            if let child = node.edges[idx].child {
                node = child
                path.edgeIndices.append(idx)
                path.nodes.append(child)
                continue
            }
            guard nodeCount < settings.maxNodes else {
                node.edges[idx].reservations -= 1
                releasePath(path)
                return nil
            }
            let s = node.state.copy()
            try! s.play(s.toMove, node.edges[idx].move, assumeLegal: true)  // moves come from the legal mask
            let child = SearchNode(state: s)
            node.edges[idx].child = child
            nodeCount += 1
            path.edgeIndices.append(idx)
            path.nodes.append(child)
            return path
        }
    }

    private func selectEdge(_ node: SearchNode) -> Int? {
        guard !node.edges.isEmpty else { return nil }
        let parentIsWhite = node.toMove == .white
        let n = Double(max(1, node.visits))
        let sqrtN = n.squareRoot()
        let parentQ = parentIsWhite ? node.nnWhiteValue : -node.nnWhiteValue
        var best = -Double.infinity
        var bestIdx = -1
        for (i, e) in node.edges.enumerated() {
            let visited = e.visits
            var q: Double
            if let c = e.child, c.visits > 0 {
                let w = c.meanWhiteValue
                q = parentIsWhite ? w : -w
            } else {
                q = parentQ - settings.fpuReduction
            }
            if e.reservations > 0 {
                q = (q * Double(visited) + settings.virtualLossValue * Double(e.reservations)) / Double(visited + e.reservations)
            }
            let u = settings.cpuct * e.prior * sqrtN / Double(1 + visited + e.reservations)
            let score = q + u
            if score > best { best = score; bestIdx = i }
        }
        return bestIdx >= 0 ? bestIdx : nil
    }

    private func releasePath(_ path: SearchPath) {
        for (i, parent) in path.nodes.dropLast().enumerated() {
            parent.edges[path.edgeIndices[i]].reservations -= 1
        }
    }

    private func releaseAll(_ batch: [SearchPath]) {
        for p in batch { releasePath(p) }
    }

    /// Evaluates and backs up one leaf batch, tagged with the `generation` captured by the caller
    /// when it issued this request. See `run(visits:deadline:)` for why: the `await` below is a
    /// reentrancy point on this actor, so `generation` may have moved by the time it returns.
    private func evaluateBatch(_ batch: [SearchPath], generation gen: Int) async throws {
        let need = batch.filter { $0.nodes.last!.terminal == nil && !$0.nodes.last!.evaluated }
        if !need.isEmpty {
            let snaps = need.map { $0.nodes.last!.state.snapshot() }
            let evals = try await evaluator.evaluate(snaps)
            guard gen == generation else {
                // Late result from a previous generation (`makeMove`/`reset` ran while this
                // request was in flight): drop it before touching a single node. Reservations on
                // this path are released so a subtree that got reused into the new generation is
                // not left with a permanent virtual loss.
                releaseAll(batch)
                return
            }
            guard evals.count == need.count else { throw SearchError(message: "evaluator returned \(evals.count) results for \(need.count) leaves") }
            for (path, (snap, e)) in zip(need, zip(snaps, evals)) {
                let leaf = path.nodes.last!
                guard !leaf.evaluated else { continue }
                if case let .rollout(count, maxMoves, weightNetwork) = settings.valueSource.mode {
                    // `ValueSource.rollout` needs the async evaluator (for playout policy), so it
                    // can't run inside the synchronous `expand` below. It's computed as a pure
                    // function of `leaf.state`/`e` first (no node mutation at all) — every
                    // `evaluator.evaluate` call inside is itself a reentrancy point on this actor
                    // (same as the call above), so `expand` (which marks the node `evaluated` and
                    // is what makes it un-retryable) is deferred until *after* the rollout
                    // finishes and the generation is re-checked. A mid-rollout `makeMove`/`reset`
                    // therefore drops this leaf exactly like the guard above (never expanded, so a
                    // later visit retries it cleanly) instead of leaving it half-expanded.
                    let blended = try await rolloutBlendedWhiteValue(leafState: leaf.state, networkEvaluation: e, snapshot: snap, count: count, maxMoves: maxMoves, weightNetwork: weightNetwork)
                    guard gen == generation else {
                        releaseAll(batch)
                        return
                    }
                    expand(leaf, snapshot: snap, evaluation: e, overrideWhiteValue: blended)
                } else {
                    expand(leaf, snapshot: snap, evaluation: e)
                }
            }
        }
        for path in batch {
            let leaf = path.nodes.last!
            let (v, s): (Double, Double) = leaf.terminal.map { (Double($0.value), Double($0.score)) } ?? (leaf.nnWhiteValue, leaf.nnWhiteScore)
            backup(path, whiteValue: v, whiteScore: s)
        }
    }

    private func expand(_ node: SearchNode, snapshot: PositionSnapshot, evaluation e: LogicEvaluation, overrideWhiteValue: Double? = nil) {
        // `overrideWhiteValue` is how `ValueSource.rollout` plugs in: the caller already computed
        // the network/rollout blend (`rolloutBlendedWhiteValue`) as a pure function before this
        // node was touched at all, so `expand` just needs to use it instead of the plain
        // `.network` conversion below. `node.rawExpected`/`rawWDL` are always the untouched raw
        // network output regardless of value source, same as every other mode.
        let w = EvaluationAdapter.toWhite(e, snapshot: snapshot, valueSource: overrideWhiteValue != nil ? .network : settings.valueSource)
        node.evaluated = true
        node.nnWhiteValue = overrideWhiteValue ?? Double(w.whiteWinValue)
        node.nnWhiteScore = Double(w.whiteScoreMean)
        node.rawExpected = Double(e.expectedResult)
        node.rawWDL = e.winDrawLoss
        node.stateFingerprint = snapshot.fingerprint
        let S = snapshot.boardSize
        var edges: [SearchNode.Edge] = []
        for i in 0 ... (S * S) where snapshot.legal[i] == 1 {
            edges.append(SearchNode.Edge(index: i, move: try! Coordinates.move(fromIndex: i, size: S), prior: Double(e.policy[i])))
        }
        node.edges = edges  // ascending policy index → deterministic tie-break
    }

    /// Runs `count` playouts from `leafState` in lock-step (docs/implementation-status.md
    /// 2026-09-10 §4-5), as a pure computation that never touches the search tree (see the call
    /// site's comment on why): every step evaluates every still-live playout line's policy in one
    /// `evaluator.evaluate` call (a line whose game has already ended by two passes drops out of
    /// that call), then samples one move per line (temperature 1 over legal moves, pass included)
    /// and plays it. A line not finished after `maxMoves` steps is scored as-is — an approximation
    /// — via `history.endAndScoreGameNow` on its own copy; this is tracked in
    /// `rolloutPlayoutsHitMaxMoves` for `RolloutDiagnostics` (updated here regardless of whether
    /// the caller ends up discarding the result to a generation race — the diagnostics are
    /// best-effort logging, not part of the tree's correctness). `value_rollout` (white
    /// perspective) is the mean of win=1/draw=0.5/loss=0 over the `count` lines; the result is
    /// `2 * (weightNetwork * e_nn(white) + (1 - weightNetwork) * value_rollout) - 1`, matching the
    /// `[-1, 1]` white-value scale `SearchNode.nnWhiteValue` uses everywhere else.
    private func rolloutBlendedWhiteValue(
        leafState: GameState, networkEvaluation e: LogicEvaluation, snapshot: PositionSnapshot, count: Int, maxMoves: Int, weightNetwork: Float
    ) async throws -> Double {
        let eNNWhiteExpected = Double(EvaluationAdapter.toWhite(e, snapshot: snapshot, valueSource: .network).whiteExpected)
        guard count > 0 else { return 2 * eNNWhiteExpected - 1 }
        rolloutLeaves += 1

        let lines = (0 ..< count).map { _ in leafState.copy() }
        var finished = [Bool](repeating: false, count: count)
        var step = 0
        while step < maxMoves {
            let active = (0 ..< count).filter { !finished[$0] }
            if active.isEmpty { break }
            let snaps = active.map { lines[$0].snapshot() }
            let evals = try await evaluator.evaluate(snaps)
            guard evals.count == active.count else {
                throw SearchError(message: "evaluator returned \(evals.count) results for \(active.count) rollout playouts")
            }
            for (k, idx) in active.enumerated() {
                let snap = snaps[k]
                let move = Self.samplePlayoutMove(policy: evals[k].policy, boardSize: snap.boardSize, rng: &rolloutRNG)
                try! lines[idx].play(snap.toMove, move, assumeLegal: true)  // sampled only from `s.legal`-masked policy mass
                if lines[idx].history.isGameFinished { finished[idx] = true }
            }
            step += 1
        }

        var winSum: Double = 0
        for i in 0 ..< count {
            rolloutPlayouts += 1
            if !finished[i] {
                rolloutPlayoutsHitMaxMoves += 1
                lines[i].history.endAndScoreGameNow(lines[i].board)  // approximation: scores the as-is board (docs task §3)
            }
            guard let outcome = lines[i].exactWhiteOutcome else {
                throw SearchError(message: "rollout playout did not produce a scored outcome")
            }
            winSum += (Double(outcome.value) + 1) / 2   // win=1, draw=0.5, loss=0, white perspective
        }
        let valueRollout = winSum / Double(count)
        let blended = Double(weightNetwork) * eNNWhiteExpected + Double(1 - weightNetwork) * valueRollout
        return 2 * blended - 1
    }

    /// Samples one move from `policy` ([S*S+1], masked softmax over legal moves so it already sums
    /// to 1 over legal moves and is exactly 0 elsewhere — docs/spec/01-network.md §5) using `rng`.
    /// Falls back to the last positive-mass index on floating-point rounding so this always returns
    /// a legal move.
    private static func samplePlayoutMove(policy: [Float], boardSize: Int, rng: inout SplitMix64) -> MoveCoord {
        let r = Double.random(in: 0 ..< 1, using: &rng)
        var cumulative: Double = 0
        var lastPositive = policy.count - 1
        for i in policy.indices {
            if policy[i] > 0 { lastPositive = i }
            cumulative += Double(policy[i])
            if r < cumulative { return try! Coordinates.move(fromIndex: i, size: boardSize) }
        }
        return try! Coordinates.move(fromIndex: lastPositive, size: boardSize)
    }

    private func backup(_ path: SearchPath, whiteValue: Double, whiteScore: Double) {
        for node in path.nodes {
            node.visits += 1
            node.whiteValueSum += whiteValue
            node.whiteScoreSum += whiteScore
        }
        for (i, parent) in path.nodes.dropLast().enumerated() {
            let edge = path.edgeIndices[i]
            parent.edges[edge].visits += 1
            parent.edges[edge].reservations -= 1
        }
    }

    private func chooseMove() -> MoveCoord {
        guard !root.edges.isEmpty else { return .pass }
        var best = root.edges[0]
        for e in root.edges.dropFirst() {
            if e.visits > best.visits || (e.visits == best.visits && e.prior > best.prior) { best = e }
        }
        return best.move
    }

    private func result(chosen: MoveCoord) throws -> SearchResult {
        let toMove = root.toMove
        let sign: Double = toMove == .white ? 1 : -1
        func expected(_ whiteValue: Double) -> Double { (sign * whiteValue + 1) / 2 }
        var cands: [MoveCandidate] = []
        let sorted = root.edges.enumerated().sorted { a, b in
            if a.element.visits != b.element.visits { return a.element.visits > b.element.visits }
            if a.element.prior != b.element.prior { return a.element.prior > b.element.prior }
            return a.element.index < b.element.index
        }
        for (order, (_, e)) in sorted.enumerated() {
            let child = e.child
            let wv = child.map(\.meanWhiteValue) ?? root.nnWhiteValue
            let ws = child.map(\.meanWhiteScore) ?? root.nnWhiteScore
            cands.append(MoveCandidate(move: e.move, index: e.index, visits: e.visits, prior: Float(e.prior), expectedScore: expected(wv),
                                       scoreLead: sign * ws, order: order, pv: pv(from: e, maxLength: 16)))
        }
        return SearchResult(move: chosen, rootVisits: root.visits, candidates: cands, rootRawExpected: root.rawExpected,
                            searchExpected: expected(root.meanWhiteValue), searchScoreLead: sign * root.meanWhiteScore,
                            rootRawWinDrawLoss: root.rawWDL, toMove: toMove, modelHash: modelHash)
    }

    /// Principal variation: follow max-visit children (ties → prior, index) up to `maxLength`.
    private func pv(from edge: SearchNode.Edge, maxLength: Int) -> [MoveCoord] {
        var out = [edge.move]
        var node = edge.child
        while let n = node, out.count < maxLength, !n.edges.isEmpty {
            let best = n.edges.max { a, b in
                if a.visits != b.visits { return a.visits < b.visits }
                if a.prior != b.prior { return a.prior < b.prior }
                return a.index > b.index
            }!
            guard best.visits > 0 else { break }
            out.append(best.move)
            node = best.child
        }
        return out
    }
}
