import Foundation
import IchiGoCore

/// Coordinate conventions (docs/spec/01-network.md §1).
/// Point index `p = y*S + x`, `x=0` left, `y=0` top. GTP: `x=0 → A`, letter `I` skipped, row = `S-y`.
/// SGF `aa` = top-left. Pass policy index = `S*S`.
public enum CoordinateError: Error, Equatable, CustomStringConvertible {
    case invalidGTP(String)
    case invalidSGF(String)
    case outOfBoard(x: Int, y: Int, size: Int)

    public var description: String {
        switch self {
        case let .invalidGTP(s): "invalid GTP coordinate \(s)"
        case let .invalidSGF(s): "invalid SGF coordinate \(s)"
        case let .outOfBoard(x, y, size): "(\(x),\(y)) is outside a \(size)x\(size) board"
        }
    }
}

/// A move on a square board: a point or pass.
public enum MoveCoord: Sendable, Equatable, Hashable {
    case pass
    case point(x: Int, y: Int)

    public var isPass: Bool { if case .pass = self { return true } else { return false } }
}

public enum Coordinates {
    static let gtpLetters = Array("ABCDEFGHJKLMNOPQRSTUVWXYZ")

    public static func index(x: Int, y: Int, size: Int) -> Int { y * size + x }
    public static func passIndex(size: Int) -> Int { size * size }

    public static func index(of move: MoveCoord, size: Int) -> Int {
        switch move {
        case .pass: passIndex(size: size)
        case let .point(x, y): index(x: x, y: y, size: size)
        }
    }

    public static func move(fromIndex i: Int, size: Int) throws -> MoveCoord {
        if i == size * size { return .pass }
        guard i >= 0, i < size * size else { throw CoordinateError.outOfBoard(x: i, y: -1, size: size) }
        return .point(x: i % size, y: i / size)
    }

    /// GTP vertex string. `(0,0)` on 9x9 → `A9`; `(8,8)` → `J1`.
    public static func gtpString(_ move: MoveCoord, size: Int) -> String {
        switch move {
        case .pass: "pass"
        case let .point(x, y): "\(gtpLetters[x])\(size - y)"
        }
    }

    /// Parses a GTP vertex. Rejects the letter `I`, out-of-range rows and columns. Case-insensitive.
    public static func parseGTP(_ text: String, size: Int) throws -> MoveCoord {
        let s = text.trimmingCharacters(in: .whitespaces).uppercased()
        if s == "PASS" { return .pass }
        guard let first = s.first, s.count >= 2 else { throw CoordinateError.invalidGTP(text) }
        guard let x = gtpLetters.firstIndex(of: first) else { throw CoordinateError.invalidGTP(text) }
        guard let row = Int(s.dropFirst()), row >= 1, row <= size, x < size else { throw CoordinateError.invalidGTP(text) }
        return .point(x: x, y: size - row)
    }

    /// SGF point. Empty string, or `tt` on boards ≤ 19, means pass.
    public static func parseSGF(_ text: String, size: Int) throws -> MoveCoord {
        if text.isEmpty { return .pass }
        let bytes = Array(text.utf8)
        guard bytes.count == 2, bytes.allSatisfy({ $0 >= 97 && $0 <= 122 }) else { throw CoordinateError.invalidSGF(text) }
        let x = Int(bytes[0] - 97)
        let y = Int(bytes[1] - 97)
        if x == 19, y == 19, size <= 19 { return .pass }
        guard x < size, y < size else { throw CoordinateError.invalidSGF(text) }
        return .point(x: x, y: y)
    }

    public static func sgfString(_ move: MoveCoord, size: Int) -> String {
        switch move {
        case .pass: ""
        case let .point(x, y): String(UnicodeScalar(UInt8(97 + x))) + String(UnicodeScalar(UInt8(97 + y)))
        }
    }

    /// Converts to the Core `Loc` encoding of a board with side `size`.
    public static func loc(_ move: MoveCoord, size: Int) -> Loc {
        switch move {
        case .pass: Board.passLoc
        case let .point(x, y): Location.getLoc(x, y, size)
        }
    }

    public static func move(fromLoc loc: Loc, size: Int) -> MoveCoord {
        if loc == Board.passLoc { return .pass }
        return .point(x: Location.getX(loc, size), y: Location.getY(loc, size))
    }
}
