import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import LogicModel

/// GTP state machine (docs/spec/03-engine.md §7). One `GTPEngine` owns the current game, the
/// per-size evaluators and the search. `handle(line:)` returns the full GTP response text
/// (including the trailing blank line); all diagnostics go to `log` (stderr), never stdout.
///
/// v1 limits: resign off, kata-genmove_analyze reports the root summary with the best candidate's
/// PV (child scores are root summaries — docs/spec/03-engine.md §9). Clock control (docs/spec/
/// 03-engine.md §8): `time_settings` accepts sudden death only (byo-yomi is rejected outright);
/// with no `time_settings` at all, genmove keeps the fixed `Config.visits` cap and no deadline.
/// Once `time_settings` has been given, every genmove computes a per-move budget from the clock
/// (`TimeManager.budget`) and runs the search under a `DeadlineController` watchdog, so a slow or
/// hung evaluator still returns a legal move by the deadline (docs/spec/03-engine.md §8).
public actor GTPEngine {
    public struct Config: Sendable {
        public var visits: Int = 100
        public var searchSettings = SearchSettings()
        public var defaultBoardSize: Int = 9
        public init() {}
    }

    public struct ModelSlot: Sendable {
        public let evaluator: any PositionEvaluating
        public let modelHash: String
        public init(evaluator: any PositionEvaluating, modelHash: String) { self.evaluator = evaluator; self.modelHash = modelHash }
    }

    public static let name = "IchiGo"
    public static let version = "0.1.0"
    public static let protocolVersion = "2"
    public static let commands = [
        "protocol_version", "name", "version", "known_command", "list_commands", "boardsize", "clear_board", "komi", "play",
        "genmove", "time_settings", "time_left", "undo", "showboard", "final_score", "quit", "kata-genmove_analyze",
    ]

    private let models: [Int: ModelSlot]
    private var config: Config
    private let log: @Sendable (String) -> Void
    private let clock: any MonotonicClock
    private var boardSize: Int
    private var komi: Float
    private var game: GameState
    private var search: Search?
    private var timeLeft: [Player: Double] = [:]
    private var mainTime: Double?
    public private(set) var quitRequested = false

    public init(models: [Int: ModelSlot], config: Config = Config(), clock: any MonotonicClock = SystemMonotonicClock(), log: @escaping @Sendable (String) -> Void) throws {
        guard !models.isEmpty else { throw SearchError(message: "at least one model is required") }
        self.models = models
        self.config = config
        self.clock = clock
        self.log = log
        let size = models[config.defaultBoardSize] != nil ? config.defaultBoardSize : models.keys.sorted()[0]
        boardSize = size
        komi = size == 9 ? IchiGoRules.defaultKomi9 : IchiGoRules.defaultKomi19
        game = try GameState(boardSize: size, komi: komi)
        // Echoes the effective CLI-parsed config (docs/spec/03-engine.md §3-4 "値ソース") once at
        // startup, so an operator diffing two `gtp` processes' logs (e.g. an A/B match) can
        // confirm which value source each side actually ran with.
        log("config: value-source=\(config.searchSettings.valueSource.logDescription) visits=\(config.visits)")
    }

    // MARK: - protocol plumbing

    public struct Parsed { let id: String?; let command: String; let args: [String] }

    /// Strips comments/CR, splits the optional numeric id. Returns nil for blank lines.
    public static func parse(_ raw: String) -> Parsed? {
        var line = raw.replacingOccurrences(of: "\r", with: "")
        if let hash = line.firstIndex(of: "#") { line = String(line[..<hash]) }
        line = line.replacingOccurrences(of: "\t", with: " ")
        let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard !parts.isEmpty else { return nil }
        var id: String? = nil
        var rest = parts
        if let first = parts.first, Int(first) != nil { id = first; rest = Array(parts.dropFirst()) }
        guard let cmd = rest.first else { return nil }
        return Parsed(id: id, command: cmd.lowercased(), args: Array(rest.dropFirst()))
    }

    static func success(_ id: String?, _ payload: String) -> String {
        "=\(id ?? "")\(payload.isEmpty || payload.hasPrefix("\n") ? payload : " " + payload)\n\n"
    }
    static func failure(_ id: String?, _ message: String) -> String { "?\(id ?? "") \(message)\n\n" }

    /// Handles one input line. Empty/comment lines yield an empty string (no response).
    public func handle(line: String) async -> String {
        guard let p = Self.parse(line) else { return "" }
        do {
            let payload = try await dispatch(p)
            return Self.success(p.id, payload)
        } catch let e as GTPError {
            return Self.failure(p.id, e.message)
        } catch {
            log("error: \(error)")
            return Self.failure(p.id, "internal error: \(error)")
        }
    }

    struct GTPError: Error { let message: String }

    // MARK: - commands

    private func dispatch(_ p: Parsed) async throws -> String {
        switch p.command {
        case "protocol_version": return Self.protocolVersion
        case "name": return Self.name
        case "version": return Self.version
        case "known_command": return Self.commands.contains(p.args.first?.lowercased() ?? "") ? "true" : "false"
        case "list_commands": return Self.commands.joined(separator: "\n")
        case "quit": quitRequested = true; return ""
        case "boardsize":
            guard let s = p.args.first.flatMap(Int.init) else { throw GTPError(message: "boardsize not an integer") }
            guard s == 9 || s == 19 else { throw GTPError(message: "unacceptable size") }
            guard let slot = models[s] else { throw GTPError(message: "no model loaded for board size \(s)") }
            let newKomi = komi
            let newGame = try GameState(boardSize: s, komi: newKomi)
            try await slot.evaluator.preWarm(size: s)
            boardSize = s; game = newGame; search = nil
            return ""
        case "clear_board":
            game = try GameState(boardSize: boardSize, komi: komi); search = nil; return ""
        case "komi":
            guard let k = p.args.first.flatMap(Float.init), k.isFinite else { throw GTPError(message: "komi not a number") }
            guard IchiGoRules.komiRange.contains(k), Rules.komiIsIntOrHalfInt(k) else { throw GTPError(message: "komi must be an integer or half-integer in [-150,150]") }
            try game.setKomi(k); komi = k; search = nil; return ""
        case "play":
            guard p.args.count >= 2, let player = Self.color(p.args[0]) else { throw GTPError(message: "invalid color or coordinate") }
            let move: MoveCoord
            do { move = try Coordinates.parseGTP(p.args[1], size: boardSize) } catch { throw GTPError(message: "invalid color or coordinate") }
            guard player == game.toMove else { throw GTPError(message: "illegal move: it is \(game.toMove == .black ? "black" : "white")'s turn") }
            guard game.isLegal(player, move) else { throw GTPError(message: "illegal move") }
            try game.play(player, move)
            if let s = search { try await s.makeMove(move) }
            return ""
        case "genmove":
            guard let player = p.args.first.flatMap(Self.color) else { throw GTPError(message: "invalid color") }
            let (move, _) = try await generateMove(for: player)
            return Coordinates.gtpString(move, size: boardSize)
        case "kata-genmove_analyze":
            guard let player = p.args.first.flatMap(Self.color) else { throw GTPError(message: "invalid color") }
            let (move, result) = try await generateMove(for: player)
            return Self.analysisPayload(result, move: move, size: boardSize)
        case "undo":
            guard game.moveNumber > 0 else { throw GTPError(message: "cannot undo") }
            try game.undo(); search = nil; return ""
        case "showboard": return "\n" + game.board.printBoard()
        case "final_score":
            let g = game.copy()
            if !g.history.isGameFinished { g.history.endAndScoreGameNow(g.board); log("final_score: game not finished on the board; score is provisional") }
            guard let o = g.exactWhiteOutcome else { throw GTPError(message: "cannot score") }
            if o.whiteMinusBlack == 0 { return "0" }
            return o.whiteMinusBlack > 0 ? "W+\(fmt(o.whiteMinusBlack))" : "B+\(fmt(-o.whiteMinusBlack))"
        case "time_settings":
            guard p.args.count >= 3, let main = Double(p.args[0]), let byo = Double(p.args[1]), let stones = Int(p.args[2]) else { throw GTPError(message: "syntax error") }
            do { try TimeManager.validateSuddenDeath(byo: byo, stones: stones) } catch { throw GTPError(message: "\(error)") }
            mainTime = main; timeLeft = [.black: main, .white: main]; return ""
        case "time_left":
            guard p.args.count >= 3, let c = Self.color(p.args[0]), let t = Double(p.args[1]) else { throw GTPError(message: "syntax error") }
            timeLeft[c] = t; return ""
        default:
            throw GTPError(message: "unknown command")
        }
    }

    /// Shared by genmove and kata-genmove_analyze: search → legality re-check → commit → respond.
    /// When `time_settings` was given, this computes a per-move deadline from the clock
    /// (docs/spec/03-engine.md §8) and runs the search under `DeadlineController`, which guarantees
    /// exactly one move is committed to the search tree even if the deadline races the search's own
    /// completion. With no `time_settings`, the search runs to the fixed `config.visits` with no
    /// deadline (unchanged v1 default behaviour).
    private func generateMove(for player: Player) async throws -> (MoveCoord, SearchResult?) {
        guard let slot = models[boardSize] else { throw GTPError(message: "no model for size \(boardSize)") }
        guard player == game.toMove else { throw GTPError(message: "it is \(game.toMove == .black ? "black" : "white")'s turn") }
        if game.history.isGameFinished {
            log("genmove after game end: passing without state change")
            return (.pass, nil)
        }
        if search == nil {
            search = try Search(evaluator: slot.evaluator, modelHash: slot.modelHash, settings: config.searchSettings, initial: game.record, clock: clock)
        }
        let s = search!

        let start = await clock.now()
        var deadline: Double?
        var remaining: Double?
        var budget: Double?
        var visitsTarget = config.visits
        if mainTime != nil {
            // Server `time_left` values take precedence over local accounting (docs/spec/03-engine
            // .md §8); `timeLeft[player]` already holds the latest one `time_left` reported, or the
            // `time_settings` main time if none has arrived yet for this colour.
            let r = timeLeft[player] ?? 0
            remaining = r
            let b = TimeManager.budget(remaining: r, moveNumber: game.moveNumber, boardSize: boardSize)
            budget = b
            deadline = start + b
            // The deadline is now the binding stop condition; let the search run as far as the
            // node budget allows instead of an unrelated small fixed-visits cap.
            visitsTarget = config.searchSettings.maxNodes
        }

        let outcome = try await DeadlineController.run(search: s, visits: visitsTarget, deadline: deadline, clock: clock)
        guard game.isLegal(player, outcome.move) else { throw GTPError(message: "search produced an illegal move \(outcome.move)") }
        try game.play(player, outcome.move)

        if let remaining {
            let elapsed = await clock.now() - start
            timeLeft[player] = max(0, remaining - elapsed)
            log("genmove clock: remaining=\(String(format: "%.3f", remaining)) budget=\(String(format: "%.3f", budget ?? 0)) actual=\(String(format: "%.3f", elapsed)) timedOut=\(outcome.timedOut)")
        }
        if let result = outcome.result {
            // `rawNN` is e_nn (the network's own raw expected result, untouched by valueSource);
            // `expected(draw=0.5)` is the search's backed-up value, which reflects `valueSource`
            // when it is not `.network` (docs/spec/03-engine.md §3-4 "値ソース").
            log("genmove \(Coordinates.gtpString(outcome.move, size: boardSize)) visits=\(result.rootVisits) expected(draw=0.5)=\(String(format: "%.3f", result.searchExpected)) rawNN=\(String(format: "%.3f", result.rootRawExpected)) valueSource=\(config.searchSettings.valueSource.logDescription) model=\(result.modelHash.prefix(12))")
        } else {
            log("genmove \(Coordinates.gtpString(outcome.move, size: boardSize)) fallback (deadline watchdog fired) model=\(slot.modelHash.prefix(12))")
        }
        return (outcome.move, outcome.result)
    }

    static func analysisPayload(_ r: SearchResult?, move: MoveCoord, size: Int) -> String {
        var lines: [String] = []
        if let r, let best = r.candidates.first {
            let pv = best.pv.map { Coordinates.gtpString($0, size: size) }.joined(separator: " ")
            lines.append("info move \(Coordinates.gtpString(best.move, size: size)) visits \(best.visits) winrate \(String(format: "%.6f", r.searchExpected)) scoreLead \(String(format: "%.6f", r.searchScoreLead)) prior \(String(format: "%.6f", best.prior)) order 0 pv \(pv)")
        }
        lines.append("play \(Coordinates.gtpString(move, size: size))")
        return "\n" + lines.joined(separator: "\n")
    }

    static func color(_ s: String) -> Player? {
        switch s.lowercased() {
        case "b", "black": .black
        case "w", "white": .white
        default: nil
        }
    }

    private func fmt(_ v: Float) -> String { v == v.rounded() ? String(Int(v)) : String(format: "%.1f", v) }
}

/// Runs the engine over stdin/stdout until `quit` or EOF. Stdout carries GTP responses only.
public enum GTPLoop {
    public static func run(engine: GTPEngine) async {
        while let line = readLine(strippingNewline: true) {
            let resp = await engine.handle(line: line)
            if !resp.isEmpty {
                FileHandle.standardOutput.write(resp.data(using: .utf8)!)
            }
            if await engine.quitRequested { break }
        }
    }
}
