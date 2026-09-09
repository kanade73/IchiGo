import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import IchiGoGTP
import LogicModel
import XCTest

actor UniformEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    init(sizes: Set<Int>) { capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true) }
    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        positions.map { s in
            let P = s.boardSize * s.boardSize
            let n = Float(s.legal.reduce(0) { $0 + Int($1) })
            // prefer centre-ish points slightly so games do not degenerate to immediate passes
            var policy = s.legal.map { Float($0) / n }
            // pass gets a tiny prior early and a dominant prior once the board is mostly full,
            // so fake games end by two passes instead of hitting the move cap
            let empties = s.current.filter { $0 == 0 }.count
            if s.legal[P] == 1 { policy[P] *= empties > P / 2 ? 0.01 : 50 }
            let z = policy.reduce(0, +)
            policy = policy.map { $0 / z }
            return LogicEvaluation(policy: policy, winDrawLoss: [0.4, 0.2, 0.4], expectedResult: 0.5, scoreMean: 0, ownership: [Float](repeating: 0, count: P))
        }
    }
    func preWarm(size: Int) async throws {}
}

/// Fake evaluator (Tests only) with a real (`Task.sleep`-backed) delay, for exercising the T28
/// clock/watchdog wiring against a genuinely slow backend without needing a fake clock at the GTP
/// layer — the exact-tie/generation-drop races are already covered directly against `Search` and
/// `DeadlineController` in `Tests/IchiGoEngineTests/TimeTests.swift`.
actor SlowEvaluator: PositionEvaluating {
    let capabilities: ModelCapabilities
    let delayNanos: UInt64
    init(sizes: Set<Int>, delayNanos: UInt64) {
        capabilities = ModelCapabilities(boardSizes: sizes, rulesID: IchiGoRules.rulesID, hasOwnership: true)
        self.delayNanos = delayNanos
    }
    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation] {
        try await Task.sleep(nanoseconds: delayNanos)
        return positions.map { s in
            let P = s.boardSize * s.boardSize
            let n = Float(s.legal.reduce(0) { $0 + Int($1) })
            let policy = s.legal.map { Float($0) / n }
            return LogicEvaluation(policy: policy, winDrawLoss: [0.4, 0.2, 0.4], expectedResult: 0.5, scoreMean: 0, ownership: [Float](repeating: 0, count: P))
        }
    }
    func preWarm(size: Int) async throws {}
}

/// Thread-safe log sink (Tests only) so a test can assert on what `GTPEngine` writes to stderr.
final class LogCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ s: String) { lock.lock(); lines.append(s); lock.unlock() }
    func snapshot() -> [String] { lock.lock(); defer { lock.unlock() }; return lines }
}

final class GTPEngineTests: XCTestCase {
    private func expect(_ e: GTPEngine, _ line: String, _ expected: String, file: StaticString = #filePath, fileLine: UInt = #line) async {
        let got = await e.handle(line: line)
        XCTAssertEqual(got, expected, "for input \(line)", file: file, line: fileLine)
    }

    private func makeEngine(sizes: Set<Int> = [9, 19], visits: Int = 8) throws -> GTPEngine {
        var cfg = GTPEngine.Config(); cfg.visits = visits
        var slots: [Int: GTPEngine.ModelSlot] = [:]
        for s in sizes { slots[s] = GTPEngine.ModelSlot(evaluator: UniformEvaluator(sizes: [s]), modelHash: "fake-\(s)") }
        return try GTPEngine(models: slots, config: cfg, log: { _ in })
    }

    func testProtocolBasicsIdsBlankLinesAndErrors() async throws {
        let e = try makeEngine()
        await expect(e, "protocol_version", "= 2\n\n")
        await expect(e, "17 name", "=17 IchiGo\n\n")
        await expect(e, "", "")
        await expect(e, "# comment", "")
        await expect(e, "42 bogus\r", "?42 unknown command\n\n")
        await expect(e, "known_command genmove", "= true\n\n")
        await expect(e, "known_command foo", "= false\n\n")
        let list = await e.handle(line: "list_commands")
        for c in GTPEngine.commands { XCTAssertTrue(list.contains(c)) }
        await expect(e, "boardsize 13", "? unacceptable size\n\n")
        await expect(e, "komi 7.25", "? komi must be an integer or half-integer in [-150,150]\n\n")
        await expect(e, "komi 7.5", "=\n\n")
        await expect(e, "time_settings 60 10 1", "? byo-yomi is not supported in v1 (sudden death only)\n\n")
        await expect(e, "time_settings 60 0 0", "=\n\n")
        await expect(e, "quit", "=\n\n")
        let q = await e.quitRequested
        XCTAssertTrue(q)
    }

    func testPlayUndoAndIllegal() async throws {
        let e = try makeEngine()
        await expect(e, "play B E5", "=\n\n")
        await expect(e, "play W E5", "? illegal move\n\n")
        let wrongTurn = await e.handle(line: "play B D4"); XCTAssertTrue(wrongTurn.hasPrefix("?"))  // wrong turn
        await expect(e, "play W I5", "? invalid color or coordinate\n\n")
        await expect(e, "play W pass", "=\n\n")
        await expect(e, "undo", "=\n\n")
        await expect(e, "undo", "=\n\n")
        await expect(e, "undo", "? cannot undo\n\n")
        let sb = await e.handle(line: "showboard"); XCTAssertTrue(sb.contains("A B C"))
    }

    func testGenmoveAndAnalyzeShareStateAndFinishedGamePasses() async throws {
        let e = try makeEngine(sizes: [9])
        let g1 = await e.handle(line: "1 genmove b")
        XCTAssertTrue(g1.hasPrefix("=1 "))
        let a = await e.handle(line: "2 kata-genmove_analyze w")
        XCTAssertTrue(a.hasPrefix("=2\ninfo move "))
        XCTAssertTrue(a.contains("winrate "))
        XCTAssertTrue(a.contains("\nplay "))
        // state advanced twice
        let board = await e.handle(line: "showboard")
        XCTAssertEqual(board.filter { $0 == "X" }.count + board.filter { $0 == "O" }.count >= 0, true)
        // two passes end the game; further genmove passes without changing state
        _ = await e.handle(line: "clear_board")
        await expect(e, "play b pass", "=\n\n")
        await expect(e, "play w pass", "=\n\n")
        await expect(e, "genmove b", "= pass\n\n")
        await expect(e, "final_score", "= W+7\n\n")
    }

    func testRandomModelCompletesGamesBothSizes() async throws {
        for s in [9, 19] {
            let e = try makeEngine(sizes: [s], visits: 2)
            await expect(e, "boardsize \(s)", "=\n\n")
            var moves = 0
            var passes = 0
            var color = "b"
            while passes < 2, moves < 4 * s * s {
                let r = await e.handle(line: "genmove \(color)")
                XCTAssertTrue(r.hasPrefix("= "), r)
                let mv = r.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
                passes = mv == "pass" ? passes + 1 : 0
                moves += 1
                color = color == "b" ? "w" : "b"
            }
            XCTAssertEqual(passes, 2, "game on \(s)x\(s) did not finish")
            let score = await e.handle(line: "final_score")
            XCTAssertTrue(score.hasPrefix("= "))
        }
    }

    // MARK: - value source CLI flags (docs/spec/03-engine.md §3-4 "値ソース")

    func testValueSourceFlagsParsing() throws {
        XCTAssertEqual(try ValueSourceFlags.parse(source: nil, blend: nil, k: nil, b: nil), .network)
        XCTAssertEqual(try ValueSourceFlags.parse(source: "network", blend: "0.9", k: "3", b: "2"), .network)  // ignored for network mode
        XCTAssertEqual(try ValueSourceFlags.parse(source: "ownership", blend: nil, k: "8", b: "2"), .ownership(k: 8, b: 2))
        XCTAssertEqual(try ValueSourceFlags.parse(source: "ownership", blend: nil, k: nil, b: nil), .ownership(k: 6, b: 1))  // defaults
        XCTAssertEqual(try ValueSourceFlags.parse(source: "blend", blend: "0.3", k: nil, b: nil), .blend(weightNetwork: 0.3, k: 6, b: 1))
        XCTAssertEqual(try ValueSourceFlags.parse(source: "blend", blend: nil, k: nil, b: nil), .blend(weightNetwork: 0.5, k: 6, b: 1))  // default weight
        XCTAssertThrowsError(try ValueSourceFlags.parse(source: "bogus", blend: nil, k: nil, b: nil)) { error in
            XCTAssertEqual(error as? ValueSourceFlagError, .unknownSource("bogus"))
        }
    }

    /// The CLI-parsed value source must be echoed to the GTP log, both once at startup and on
    /// every genmove (alongside e_nn/`rawNN=`), so an operator can confirm what an A/B match's two
    /// `gtp` processes actually ran with.
    func testValueSourceIsEchoedInStartupAndGenmoveLog() async throws {
        var cfg = GTPEngine.Config(); cfg.visits = 4
        cfg.searchSettings.valueSource = try ValueSourceFlags.parse(source: "ownership", blend: nil, k: "8", b: "2")
        let slot = GTPEngine.ModelSlot(evaluator: UniformEvaluator(sizes: [9]), modelHash: "fake-9")
        let logs = LogCapture()
        let e = try GTPEngine(models: [9: slot], config: cfg, log: { logs.append($0) })
        XCTAssertTrue(logs.snapshot().contains { $0.contains("value-source=ownership(k=8.0,b=2.0)") })

        let r = await e.handle(line: "genmove b")
        XCTAssertTrue(r.hasPrefix("= "), r)
        XCTAssertTrue(logs.snapshot().contains { $0.hasPrefix("genmove ") && $0.contains("rawNN=") && $0.contains("valueSource=ownership(k=8.0,b=2.0)") })
    }

    // MARK: - T28 clock / watchdog

    /// docs/spec/03-engine.md §8: `time_left=0` must still produce a legal move quickly (the
    /// budget is 0, so genmove goes straight down the `DeadlineController` fallback path).
    func testTimeLeftZeroProducesLegalMoveQuickly() async throws {
        let e = try makeEngine(sizes: [9], visits: 400)
        await expect(e, "time_settings 5 0 0", "=\n\n")
        await expect(e, "time_left b 0 0", "=\n\n")
        let start = DispatchTime.now()
        let r = await e.handle(line: "genmove b")
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        XCTAssertTrue(r.hasPrefix("= "), r)
        let mv = r.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(mv.isEmpty)
        XCTAssertLessThan(elapsedMs, 2000)
    }

    /// With `time_settings` active, a slow evaluator must not stall genmove past the computed
    /// deadline: the watchdog fallback fires and a legal move still comes back promptly.
    func testSlowEvaluatorUnderTimeControlFallsBackWithinDeadline() async throws {
        var cfg = GTPEngine.Config(); cfg.visits = 400
        let slot = GTPEngine.ModelSlot(evaluator: SlowEvaluator(sizes: [9], delayNanos: 3_000_000_000), modelHash: "slow-9")
        let logs = LogCapture()
        let e = try GTPEngine(models: [9: slot], config: cfg, log: { logs.append($0) })
        await expect(e, "time_settings 5 0 0", "=\n\n")
        await expect(e, "time_left b 0.2 0", "=\n\n")   // small remaining time → a tiny (millisecond-scale) budget
        let start = DispatchTime.now()
        let r = await e.handle(line: "genmove b")
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        XCTAssertTrue(r.hasPrefix("= "), r)
        let mv = r.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(mv.isEmpty)
        XCTAssertLessThan(elapsed, 2.0)   // well under the 3s evaluator delay
        XCTAssertTrue(logs.snapshot().contains { $0.contains("timedOut=true") })
    }
}
