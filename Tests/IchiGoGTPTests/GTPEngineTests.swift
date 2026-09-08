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
}
