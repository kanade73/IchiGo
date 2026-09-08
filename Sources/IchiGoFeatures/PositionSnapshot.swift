import Foundation
import IchiGoCore

/// The only rule set supported by v1: `cgos-area-psk-v1` (docs/spec/03-engine.md §5).
public enum IchiGoRules {
    public static let rulesID = "cgos-area-psk-v1"
    public static let defaultKomi9: Float = 7.0
    public static let defaultKomi19: Float = 7.5
    public static let komiRange: ClosedRange<Float> = -150 ... 150

    public static func make(komi: Float) throws -> Rules {
        guard komiRange.contains(komi), Rules.komiIsIntOrHalfInt(komi) else {
            throw GameStateError.invalidKomi(komi)
        }
        return Rules(
            koRule: .positional, scoringRule: .area, taxRule: .none, multiStoneSuicideLegal: false,
            hasButton: false, whiteHandicapBonusRule: .zero, friendlyPassOk: true, komi: komi
        )
    }
}

public enum GameStateError: Error, Equatable, CustomStringConvertible {
    case unsupportedBoardSize(Int)
    case invalidKomi(Float)
    case illegalMove(player: Player, move: MoveCoord, reason: String)
    case invalidInitialStone(MoveCoord)
    case wrongPlayer(expected: Player, got: Player)
    case emptyHistory

    public var description: String {
        switch self {
        case let .unsupportedBoardSize(s): "unsupported board size \(s)"
        case let .invalidKomi(k): "invalid komi \(k) (must be integer or half-integer in [-150,150])"
        case let .illegalMove(p, m, r): "illegal move \(m) by \(p): \(r)"
        case let .invalidInitialStone(m): "invalid initial stone at \(m)"
        case let .wrongPlayer(e, g): "expected \(e) to move, got \(g)"
        case .emptyHistory: "no move to undo"
        }
    }
}

/// Stone colour per point, plain value (0 empty, 1 black, 2 white), `S*S` entries `y*S+x`.
public typealias StoneLayout = [UInt8]

/// Immutable deep copy of everything the feature encoder and search need
/// (docs/spec/03-engine.md §2). Contains only value types, so it is `Sendable` without
/// `@unchecked`. Later board mutation cannot change a snapshot.
public struct PositionSnapshot: Sendable, Equatable {
    public let boardSize: Int
    public let toMove: Player
    /// White's komi (points added to white).
    public let komi: Float
    /// Moves played after the initial position (passes count).
    public let moveNumber: Int
    /// Trailing passes in the move list.
    public let consecutivePasses: Int
    /// `layouts[0]` is the current board, `layouts[t]` the board `t` moves ago; at most 8 entries
    /// (current + 7). Positions before the initial layout are absent.
    public let layouts: [StoneLayout]
    /// Most recent moves, most recent first (at most 2 kept).
    public let recentMoves: [MoveCoord]
    /// Chain liberties per point (0 for empty).
    public let liberties: [Int32]
    /// Simple-ko forbidden point index, if any.
    public let koPoint: Int?
    /// `[S*S+1]` full legality for `toMove` under the v1 rules incl. positional superko; pass last.
    public let legal: [UInt8]
    /// Whether the game is over on the board (two consecutive passes).
    public let isGameFinished: Bool
    /// Fingerprint of the complete state (board, to-move, ko, superko bans, pass state, komi).
    public let fingerprint: String

    public var current: StoneLayout { layouts[0] }
}

/// Sendable replay record of a game state (initial setup + moves). Rebuilding a `GameState` from
/// it replays every move under the v1 rules, so it can cross actor boundaries safely.
public struct GameRecord: Sendable, Equatable {
    public let boardSize: Int
    public let komi: Float
    public let initialStones: [InitialStone]
    public let initialPlayer: Player
    public let moves: [RecordedMove]

    public struct InitialStone: Sendable, Equatable { public let player: Player; public let x: Int; public let y: Int
        public init(player: Player, x: Int, y: Int) { self.player = player; self.x = x; self.y = y } }
    public struct RecordedMove: Sendable, Equatable { public let player: Player; public let move: MoveCoord
        public init(player: Player, move: MoveCoord) { self.player = player; self.move = move } }

    public init(boardSize: Int, komi: Float, initialStones: [InitialStone] = [], initialPlayer: Player = .black, moves: [RecordedMove] = []) {
        self.boardSize = boardSize; self.komi = komi; self.initialStones = initialStones; self.initialPlayer = initialPlayer; self.moves = moves
    }
}

/// Owns a Core `Board`/`BoardHistory` and records every stone layout so 7 moves of history
/// are always available (Core keeps only 6 recent boards).
public final class GameState {
    public let boardSize: Int
    public private(set) var komi: Float
    public private(set) var rules: Rules
    public private(set) var board: Board
    public private(set) var history: BoardHistory
    public private(set) var moves: [(player: Player, move: MoveCoord)] = []
    public let initialStones: [(player: Player, x: Int, y: Int)]
    public let initialPlayer: Player
    private var layouts: [StoneLayout] = []

    public init(boardSize: Int, komi: Float, initialStones: [(player: Player, x: Int, y: Int)] = [], initialPlayer: Player = .black) throws {
        guard boardSize == 9 || boardSize == 19 else { throw GameStateError.unsupportedBoardSize(boardSize) }
        self.boardSize = boardSize
        self.komi = komi
        rules = try IchiGoRules.make(komi: komi)
        board = Board(boardSize, boardSize)
        self.initialStones = initialStones
        self.initialPlayer = initialPlayer
        var placed = Set<Int>()
        for s in initialStones {
            guard s.x >= 0, s.x < boardSize, s.y >= 0, s.y < boardSize,
                  placed.insert(s.y * boardSize + s.x).inserted,
                  board.setStone(Location.getLoc(s.x, s.y, boardSize), s.player.color) else {
                throw GameStateError.invalidInitialStone(.point(x: s.x, y: s.y))
            }
        }
        history = BoardHistory(board, pla: initialPlayer, rules: rules)
        layouts = [Self.layout(of: board, size: boardSize)]
    }

    public var toMove: Player { history.presumedNextMovePla }

    public convenience init(record r: GameRecord) throws {
        try self.init(boardSize: r.boardSize, komi: r.komi, initialStones: r.initialStones.map { ($0.player, $0.x, $0.y) }, initialPlayer: r.initialPlayer)
        for m in r.moves { try play(m.player, m.move) }
    }

    public var record: GameRecord {
        GameRecord(boardSize: boardSize, komi: komi, initialStones: initialStones.map { .init(player: $0.player, x: $0.x, y: $0.y) },
                   initialPlayer: initialPlayer, moves: moves.map { .init(player: $0.player, move: $0.move) })
    }

    /// Deep copy (board, history, move list, layouts). Used by search to expand child positions.
    public func copy() -> GameState {
        let c = GameState(copying: self)
        return c
    }

    private init(copying o: GameState) {
        boardSize = o.boardSize
        komi = o.komi
        rules = o.rules
        board = o.board.copy()
        history = o.history.copy()
        moves = o.moves
        initialStones = o.initialStones
        initialPlayer = o.initialPlayer
        layouts = o.layouts
    }

    /// Exact outcome of a finished game (white perspective): +1 white win, -1 black win, 0 draw,
    /// plus white-minus-black score. `nil` while the game is not finished.
    public var exactWhiteOutcome: (value: Float, whiteMinusBlack: Float)? {
        guard history.isGameFinished, history.isScored else { return nil }
        let s = history.finalWhiteMinusBlackScore
        return (s > 0 ? 1 : (s < 0 ? -1 : 0), s)
    }
    public var moveNumber: Int { moves.count }

    static func layout(of board: Board, size: Int) -> StoneLayout {
        var out = StoneLayout(repeating: 0, count: size * size)
        for y in 0 ..< size {
            for x in 0 ..< size {
                out[y * size + x] = UInt8(board.colors[Location.getLoc(x, y, size)].rawValue)
            }
        }
        return out
    }

    /// True when `move` is a pass or an on-board point (off-board coordinates must never be
    /// mapped onto the board by `Location.getLoc`).
    private func isOnBoard(_ move: MoveCoord) -> Bool {
        guard case let .point(x, y) = move else { return true }
        return x >= 0 && x < boardSize && y >= 0 && y < boardSize
    }

    public func isLegal(_ player: Player, _ move: MoveCoord) -> Bool {
        guard isOnBoard(move) else { return false }
        return history.isLegal(board, Coordinates.loc(move, size: boardSize), player)
    }

    /// Plays a move after full legality check (suicide, simple ko, positional superko, turn order).
    public func play(_ player: Player, _ move: MoveCoord) throws {
        guard player == toMove else { throw GameStateError.wrongPlayer(expected: toMove, got: player) }
        guard isOnBoard(move) else {
            throw GameStateError.illegalMove(player: player, move: move, reason: "off-board")
        }
        let loc = Coordinates.loc(move, size: boardSize)
        guard history.isLegal(board, loc, player) else {
            throw GameStateError.illegalMove(player: player, move: move, reason: "rejected by rules (occupied, suicide, ko or superko)")
        }
        history.makeBoardMoveAssumeLegal(board, loc, player)
        moves.append((player, move))
        layouts.append(Self.layout(of: board, size: boardSize))
    }

    /// Replays from the initial position without the last move (restores history, ko, pass state).
    public func undo() throws {
        guard !moves.isEmpty else { throw GameStateError.emptyHistory }
        let kept = Array(moves.dropLast())
        board = Board(boardSize, boardSize)
        for s in initialStones { _ = board.setStone(Location.getLoc(s.x, s.y, boardSize), s.player.color) }
        history = BoardHistory(board, pla: initialPlayer, rules: rules)
        moves = []
        layouts = [Self.layout(of: board, size: boardSize)]
        for m in kept { try play(m.player, m.move) }
    }

    public func setKomi(_ newKomi: Float) throws {
        rules = try IchiGoRules.make(komi: newKomi)
        komi = newKomi
        history.setKomi(newKomi)
    }

    public func snapshot() -> PositionSnapshot {
        let S = boardSize
        let pla = toMove
        var legal = [UInt8](repeating: 0, count: S * S + 1)
        var libs = [Int32](repeating: 0, count: S * S)
        for y in 0 ..< S {
            for x in 0 ..< S {
                let loc = Location.getLoc(x, y, S)
                if history.isLegal(board, loc, pla) { legal[y * S + x] = 1 }
                if board.colors[loc] == .black || board.colors[loc] == .white {
                    libs[y * S + x] = Int32(board.getNumLiberties(loc))
                }
            }
        }
        legal[S * S] = history.isLegal(board, Board.passLoc, pla) ? 1 : 0
        let ko: Int? = board.koLoc == Board.nullLoc ? nil : Location.getY(board.koLoc, S) * S + Location.getX(board.koLoc, S)
        var consecutivePasses = 0
        for m in moves.reversed() {
            if m.move.isPass { consecutivePasses += 1 } else { break }
        }
        let recent = moves.suffix(2).reversed().map(\.move)
        let hist = Array(layouts.reversed().prefix(8))
        var banned: [Int] = []
        for i in 0 ..< (S * S) where history.superKoBanned[Location.getLoc(i % S, i / S, S)] { banned.append(i) }
        let fp = Self.fingerprint(
            layouts: layouts, toMove: pla, komi: komi, ko: ko, banned: banned,
            consecutivePasses: consecutivePasses, moveNumber: moves.count, recentMoves: Array(recent)
        )
        return PositionSnapshot(
            boardSize: S, toMove: pla, komi: komi, moveNumber: moves.count, consecutivePasses: consecutivePasses,
            layouts: hist, recentMoves: Array(recent), liberties: libs, koPoint: ko, legal: legal,
            isGameFinished: history.isGameFinished, fingerprint: fp
        )
    }

    /// SHA-256 (hex) over everything the encoder and the rules depend on: the current layout, every
    /// previous layout (which is exactly the positional-superko history), to-move, komi, simple-ko
    /// point, superko bans, pass state, move number and the recent moves. Two states with different
    /// encoded features therefore always get different fingerprints.
    static func fingerprint(
        layouts: [StoneLayout], toMove: Player, komi: Float, ko: Int?, banned: [Int],
        consecutivePasses: Int, moveNumber: Int, recentMoves: [MoveCoord]
    ) -> String {
        var data = Data()
        data.append(contentsOf: Array("ichigo-fp-v1|".utf8))
        for layout in layouts {  // oldest first, last entry is the current board
            data.append(contentsOf: layout)
            data.append(0xFF)
        }
        let moveText = recentMoves.map { m -> String in
            if case let .point(x, y) = m { return "\(x),\(y)" } else { return "pass" }
        }.joined(separator: ";")
        let text = "|pla=\(toMove.rawValue)|komi=\(komi)|ko=\(ko.map(String.init) ?? "-")|sk=\(banned)"
            + "|passes=\(consecutivePasses)|n=\(moveNumber)|recent=\(moveText)"
        data.append(contentsOf: Array(text.utf8))
        return SHA256Hex.digest(data)
    }
}
