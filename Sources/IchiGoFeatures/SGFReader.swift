import Foundation
import IchiGoCore

public enum SGFError: Error, CustomStringConvertible, Equatable {
    case malformed(String)
    case unsupported(String)
    case illegalMove(String)

    public var description: String {
        switch self {
        case let .malformed(message), let .unsupported(message), let .illegalMove(message): message
        }
    }
}

public struct SGFGame: Sendable {
    public let boardXSize: Int
    public let boardYSize: Int
    public let komi: Float
    public let rules: Rules
    public let moves: [Move]
    public let placements: [Move]
    public let playerToMove: Player?
    public let result: String?

    public func position(
        at moveNumber: Int,
        rules overrideRules: Rules? = nil
    ) throws -> (board: Board, history: BoardHistory, nextPlayer: Player) {
        guard moveNumber >= 0, moveNumber <= moves.count else {
            throw SGFError.malformed(
                "Move number \(moveNumber) is outside the SGF range 0...\(moves.count)"
            )
        }
        let onlyBlackPlacements = placements.contains(where: { $0.pla == .black })
            && placements.allSatisfy { $0.pla == .black }
        var nextPlayer: Player = if let playerToMove {
            playerToMove
        } else if onlyBlackPlacements {
            .white
        } else {
            .black
        }
        if let firstMove = moves.first { nextPlayer = firstMove.pla }

        let board = Board(boardXSize, boardYSize)
        guard board.setStonesFailIfNoLibs(placements) else {
            throw SGFError.malformed("Initial setup contains overlapping or zero-liberty stones")
        }
        var selectedRules = overrideRules ?? rules
        selectedRules.komi = overrideRules?.komi ?? komi
        let history = BoardHistory(board, pla: nextPlayer, rules: selectedRules, encorePhase: 0)
        history.setInitialTurnNumber(Int64(board.numStonesOnBoard()))

        for (index, move) in moves.prefix(moveNumber).enumerated() {
            guard history.makeBoardMoveTolerant(board, move.loc, move.pla, preventEncore: false) else {
                let coordinate = Location.toString(move.loc, boardXSize, boardYSize)
                throw SGFError.illegalMove("Illegal SGF move \(index) (\(move.pla) \(coordinate))")
            }
            nextPlayer = move.pla.opponent
        }
        return (board, history, nextPlayer)
    }
}

public enum SGFReader {
    public static func load(_ url: URL, defaultRules: Rules = .getTrompTaylorish()) throws -> SGFGame {
        let data = try Data(contentsOf: url)
        guard let string = String(data: data, encoding: .utf8) else {
            throw SGFError.malformed("SGF is not valid UTF-8")
        }
        return try parse(string, defaultRules: defaultRules)
    }

    public static func parse(_ source: String, defaultRules: Rules = .getTrompTaylorish()) throws -> SGFGame {
        var parser = Parser(source)
        let nodes = try parser.parse()
        guard let root = nodes.first else { throw SGFError.malformed("Empty SGF") }
        try validate(root: root, nodes: nodes)

        let (xSize, ySize) = try parseSize(single(root, "SZ") ?? "19")
        let komi = try parseKomi(single(root, "KM"), fallback: defaultRules.komi)
        var rules = defaultRules
        if let ruleName = single(root, "RU") {
            do {
                rules = try Rules.parseRulesWithoutKomi(ruleName, komi: komi)
            } catch {
                throw SGFError.malformed("Could not parse SGF rules '\(ruleName)': \(error)")
            }
        }
        rules.komi = komi

        var placements = [Move]()
        for value in root["AB"] ?? [] {
            try placements.append(Move(loc: parseCoordinate(value, xSize, ySize, allowPass: false), pla: .black))
        }
        for value in root["AW"] ?? [] {
            try placements.append(Move(loc: parseCoordinate(value, xSize, ySize, allowPass: false), pla: .white))
        }

        var moves = [Move]()
        for node in nodes {
            if let values = node["B"] {
                guard values.count == 1 else { throw SGFError.malformed("B must have one value") }
                try moves.append(Move(loc: parseCoordinate(values[0], xSize, ySize, allowPass: true), pla: .black))
            }
            if let values = node["W"] {
                guard values.count == 1 else { throw SGFError.malformed("W must have one value") }
                try moves.append(Move(loc: parseCoordinate(values[0], xSize, ySize, allowPass: true), pla: .white))
            }
        }
        let player = try single(root, "PL").map(parsePlayer)
        return SGFGame(
            boardXSize: xSize,
            boardYSize: ySize,
            komi: komi,
            rules: rules,
            moves: moves,
            placements: placements,
            playerToMove: player,
            result: single(root, "RE")
        )
    }

    private static func validate(root: [String: [String]], nodes: [[String: [String]]]) throws {
        if root["AE"] != nil { throw SGFError.unsupported("AE setup removals are not supported") }
        let rootOnly = Set(["SZ", "KM", "RU", "HA", "AB", "AW", "PL", "RE"])
        for (index, node) in nodes.enumerated() where index > 0 {
            if node.keys.contains(where: { rootOnly.contains($0) || $0 == "AE" }) {
                throw SGFError.unsupported("Setup, rules, size, komi, and PL are only supported on the root node")
            }
            if node["B"] != nil, node["W"] != nil {
                throw SGFError.malformed("A node cannot contain both B and W moves")
            }
        }
    }

    private static func parseSize(_ value: String) throws -> (Int, Int) {
        let parts = value.split(separator: ":", omittingEmptySubsequences: false)
        guard (1 ... 2).contains(parts.count), let x = Int(parts[0]),
              let y = parts.count == 2 ? Int(parts[1]) : x,
              x > 1, y > 1, x <= Board.maxLen, y <= Board.maxLen
        else {
            throw SGFError.malformed("Invalid board size '\(value)'")
        }
        return (x, y)
    }

    private static func parseKomi(_ value: String?, fallback: Float) throws -> Float {
        guard let value else { return fallback }
        guard let komi = Float(value), Rules.komiIsIntOrHalfInt(komi) else {
            throw SGFError.malformed("Komi must be an integer or half-integer")
        }
        return komi
    }

    private static func parseCoordinate(
        _ value: String,
        _ xSize: Int,
        _ ySize: Int,
        allowPass: Bool
    ) throws -> Loc {
        if allowPass, value.isEmpty || (value.lowercased() == "tt" && xSize <= 19 && ySize <= 19) {
            return Board.passLoc
        }
        if value.contains(":") {
            throw SGFError.unsupported("Compressed point ranges are not supported")
        }
        let characters = Array(value)
        guard characters.count == 2,
              let x = sgfCoordinate(characters[0]), let y = sgfCoordinate(characters[1]),
              x < xSize, y < ySize
        else {
            throw SGFError.malformed("Invalid or out-of-bounds SGF coordinate '\(value)'")
        }
        return Location.getLoc(x, y, xSize)
    }

    private static func sgfCoordinate(_ character: Character) -> Int? {
        guard let byte = String(character).utf8.first else { return nil }
        if byte >= 97, byte <= 122 { return Int(byte - 97) }
        if byte >= 65, byte <= 90 { return Int(byte - 65 + 26) }
        return nil
    }

    private static func parsePlayer(_ value: String) throws -> Player {
        switch value.lowercased() {
        case "b", "black": .black
        case "w", "white": .white
        default: throw SGFError.malformed("Invalid PL value '\(value)'")
        }
    }

    private static func single(_ node: [String: [String]], _ key: String) -> String? {
        node[key]?.first
    }
}

private struct Parser {
    private let characters: [Character]
    private var index = 0

    init(_ source: String) {
        characters = Array(source)
    }

    mutating func parse() throws -> [[String: [String]]] {
        skipWhitespace()
        try expect("(")
        skipWhitespace()
        var nodes = [[String: [String]]]()
        while peek() == ";" {
            index += 1
            try nodes.append(parseNode())
            skipWhitespace()
        }
        guard !nodes.isEmpty else { throw SGFError.malformed("SGF game tree has no nodes") }
        if peek() == "(" { throw SGFError.unsupported("SGF variations are not supported") }
        try expect(")")
        skipWhitespace()
        guard index == characters.count else {
            throw SGFError.unsupported("Multiple SGF game trees are not supported")
        }
        return nodes
    }

    private mutating func parseNode() throws -> [String: [String]] {
        var properties = [String: [String]]()
        while true {
            skipWhitespace()
            guard peek()?.isUppercase == true else { break }
            var identifier = ""
            while let character = peek(), character.isUppercase {
                identifier.append(character)
                index += 1
            }
            skipWhitespace()
            guard peek() == "[" else {
                throw SGFError.malformed("Property \(identifier) has no value")
            }
            var values = [String]()
            while peek() == "[" {
                try values.append(parseValue())
            }
            properties[identifier, default: []].append(contentsOf: values)
        }
        return properties
    }

    private mutating func parseValue() throws -> String {
        try expect("[")
        var value = ""
        while let character = peek() {
            index += 1
            if character == "]" { return value }
            if character == "\\" {
                guard let escaped = peek() else { throw SGFError.malformed("Dangling SGF escape") }
                index += 1
                if escaped == "\n" { continue }
                if escaped == "\r" {
                    if peek() == "\n" { index += 1 }
                    continue
                }
                value.append(escaped)
            } else {
                value.append(character)
            }
        }
        throw SGFError.malformed("Unterminated SGF property value")
    }

    private mutating func expect(_ character: Character) throws {
        guard peek() == character else { throw SGFError.malformed("Expected '\(character)'") }
        index += 1
    }

    private mutating func skipWhitespace() {
        while peek()?.isWhitespace == true {
            index += 1
        }
    }

    private func peek() -> Character? {
        index < characters.count ? characters[index] : nil
    }
}
