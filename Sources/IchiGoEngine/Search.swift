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

public actor Search {
    public let settings: SearchSettings
    private let evaluator: any PositionEvaluating
    private let modelHash: String
    private var root: SearchNode
    private var nodeCount = 1
    private var generation = 0

    public init(evaluator: any PositionEvaluating, modelHash: String, settings: SearchSettings = SearchSettings(), initial: GameRecord) throws {
        self.evaluator = evaluator
        self.modelHash = modelHash
        self.settings = settings
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

    /// Plays `move` at the root. Promotes the existing child subtree when tree reuse is enabled
    /// and the child's full-state fingerprint equals the freshly replayed position; otherwise the
    /// tree is discarded. Throws on an illegal move.
    public func makeMove(_ move: MoveCoord) throws {
        let next = root.state.copy()
        guard next.isLegal(next.toMove, move) else { throw SearchError(message: "illegal move \(move)") }
        try next.play(next.toMove, move)
        generation += 1
        if settings.treeReuse, let edge = root.edges.first(where: { $0.move == move }), let child = edge.child,
           child.state.snapshot().fingerprint == next.snapshot().fingerprint {
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

    /// Runs until the root has `visits` visits (including the root's own evaluation) or the node
    /// budget is exhausted or the task is cancelled. Returns the chosen move (max edge visits,
    /// ties → prior, then policy index).
    public func run(visits target: Int) async throws -> SearchResult {
        let gen = generation
        if root.terminal != nil { return try result(chosen: .pass) }
        if !root.evaluated {
            try await evaluateBatch([[root]])
            guard gen == generation else { throw SearchError(message: "search invalidated") }
        }
        while root.visits < target, !Task.isCancelled {
            let batch = collectBatch(limit: min(settings.leafBatch, target - root.visits))
            if batch.isEmpty { break }
            do {
                try await evaluateBatch(batch)
            } catch {
                releaseAll(batch)
                throw error
            }
            guard gen == generation else { releaseAll(batch); throw SearchError(message: "search invalidated") }
        }
        return try result(chosen: chooseMove())
    }

    private func collectBatch(limit: Int) -> [[SearchNode]] {
        var paths: [[SearchNode]] = []
        while paths.count < limit {
            guard let path = descend() else { break }
            paths.append(path)
            if let last = path.last, last.terminal != nil { continue }
        }
        return paths
    }

    /// Selects a path from the root; reserves every traversed edge. Expands one new node or
    /// stops at a terminal. Returns nil when the node budget is exhausted.
    private func descend() -> [SearchNode]? {
        var path = [root]
        var node = root
        while true {
            if node.terminal != nil { return path }
            if !node.evaluated { return path }  // freshly expanded (or reserved) leaf awaiting evaluation
            guard let idx = selectEdge(node) else { return path }
            node.edges[idx].reservations += 1
            if let child = node.edges[idx].child {
                node = child
                path.append(child)
                continue
            }
            guard nodeCount < settings.maxNodes else {
                node.edges[idx].reservations -= 1
                for (i, n) in path.enumerated().dropLast() { _ = i; _ = n }
                releasePath(path)
                return nil
            }
            let s = node.state.copy()
            try! s.play(s.toMove, node.edges[idx].move)  // moves come from the legal mask
            let child = SearchNode(state: s)
            node.edges[idx].child = child
            nodeCount += 1
            path.append(child)
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

    private func releasePath(_ path: [SearchNode]) {
        for (parent, child) in zip(path, path.dropFirst()) {
            if let i = parent.edges.firstIndex(where: { $0.child === child }) { parent.edges[i].reservations -= 1 }
        }
    }

    private func releaseAll(_ batch: [[SearchNode]]) {
        for p in batch { releasePath(p) }
    }

    private func evaluateBatch(_ batch: [[SearchNode]]) async throws {
        let need = batch.filter { $0.last!.terminal == nil && !$0.last!.evaluated }
        var evals: [LogicEvaluation] = []
        if !need.isEmpty {
            let snaps = need.map { $0.last!.state.snapshot() }
            evals = try await evaluator.evaluate(snaps)
            guard evals.count == need.count else { throw SearchError(message: "evaluator returned \(evals.count) results for \(need.count) leaves") }
            for (path, (snap, e)) in zip(need, zip(snaps, evals)) {
                let leaf = path.last!
                if !leaf.evaluated { expand(leaf, snapshot: snap, evaluation: e) }
            }
        }
        for path in batch {
            let leaf = path.last!
            let (v, s): (Double, Double) = leaf.terminal.map { (Double($0.value), Double($0.score)) } ?? (leaf.nnWhiteValue, leaf.nnWhiteScore)
            backup(path, whiteValue: v, whiteScore: s)
        }
    }

    private func expand(_ node: SearchNode, snapshot: PositionSnapshot, evaluation e: LogicEvaluation) {
        let w = EvaluationAdapter.toWhite(e, snapshot: snapshot)
        node.evaluated = true
        node.nnWhiteValue = Double(w.whiteWinValue)
        node.nnWhiteScore = Double(w.whiteScoreMean)
        node.rawExpected = Double(e.expectedResult)
        node.rawWDL = e.winDrawLoss
        let S = snapshot.boardSize
        var edges: [SearchNode.Edge] = []
        for i in 0 ... (S * S) where snapshot.legal[i] == 1 {
            edges.append(SearchNode.Edge(index: i, move: try! Coordinates.move(fromIndex: i, size: S), prior: Double(e.policy[i])))
        }
        node.edges = edges  // ascending policy index → deterministic tie-break
    }

    private func backup(_ path: [SearchNode], whiteValue: Double, whiteScore: Double) {
        for node in path {
            node.visits += 1
            node.whiteValueSum += whiteValue
            node.whiteScoreSum += whiteScore
        }
        for (parent, child) in zip(path, path.dropFirst()) {
            if let i = parent.edges.firstIndex(where: { $0.child === child }) {
                parent.edges[i].visits += 1
                parent.edges[i].reservations -= 1
            }
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
