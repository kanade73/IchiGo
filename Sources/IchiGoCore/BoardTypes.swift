public typealias Loc = Int

public enum Player: Int8, CaseIterable, Sendable {
    case black = 1
    case white = 2

    public var opponent: Player {
        self == .black ? .white : .black
    }

    public var color: Color {
        self == .black ? .black : .white
    }
}

public enum Color: Int8, Sendable {
    case empty = 0
    case black = 1
    case white = 2
    case wall = 3

    public var player: Player? {
        switch self {
        case .black: .black
        case .white: .white
        case .empty, .wall: nil
        }
    }
}

public let P_BLACK = Player.black
public let P_WHITE = Player.white
public let C_EMPTY = Color.empty
public let C_BLACK = Color.black
public let C_WHITE = Color.white
public let C_WALL = Color.wall

public struct Move: Equatable, Sendable {
    public var loc: Loc
    public var pla: Player

    public init(loc: Loc, pla: Player) {
        self.loc = loc
        self.pla = pla
    }
}

public struct StonePlacement: Equatable, Sendable {
    public var loc: Loc
    public var color: Color

    public init(loc: Loc, color: Color) {
        self.loc = loc
        self.color = color
    }
}

public enum Location {
    public static func getLoc(_ x: Int, _ y: Int, _ xSize: Int) -> Loc {
        (x + 1) + (y + 1) * (xSize + 1)
    }

    public static func getX(_ loc: Loc, _ xSize: Int) -> Int {
        loc % (xSize + 1) - 1
    }

    public static func getY(_ loc: Loc, _ xSize: Int) -> Int {
        loc / (xSize + 1) - 1
    }

    public static func getAdjacentOffsets(_ xSize: Int) -> [Int] {
        [-(xSize + 1), -1, 1, xSize + 1, -(xSize + 1) - 1, -(xSize + 1) + 1, xSize, xSize + 2]
    }

    public static func isAdjacent(_ first: Loc, _ second: Loc, _ xSize: Int) -> Bool {
        let stride = xSize + 1
        return first == second - stride || first == second - 1 || first == second + 1 || first == second + stride
    }

    public static func getMirrorLoc(_ loc: Loc, _ xSize: Int, _ ySize: Int) -> Loc {
        guard loc != Board.nullLoc, loc != Board.passLoc else { return loc }
        return getLoc(xSize - 1 - getX(loc, xSize), ySize - 1 - getY(loc, xSize), xSize)
    }

    public static func getCenterLoc(_ xSize: Int, _ ySize: Int) -> Loc {
        guard xSize % 2 == 1, ySize % 2 == 1 else { return Board.nullLoc }
        return getLoc(xSize / 2, ySize / 2, xSize)
    }

    public static func isCentral(_ loc: Loc, _ xSize: Int, _ ySize: Int) -> Bool {
        let x = getX(loc, xSize)
        let y = getY(loc, xSize)
        return x >= (xSize - 1) / 2 && x <= xSize / 2 && y >= (ySize - 1) / 2 && y <= ySize / 2
    }

    public static func isNearCentral(_ loc: Loc, _ xSize: Int, _ ySize: Int) -> Bool {
        let x = getX(loc, xSize)
        let y = getY(loc, xSize)
        return x >= (xSize - 1) / 2 - 1 && x <= xSize / 2 + 1
            && y >= (ySize - 1) / 2 - 1 && y <= ySize / 2 + 1
    }

    public static func distance(_ first: Loc, _ second: Loc, _ xSize: Int) -> Int {
        let dx = getX(second, xSize) - getX(first, xSize)
        let dy = (second - first - dx) / (xSize + 1)
        return abs(dx) + abs(dy)
    }

    public static func euclideanDistanceSquared(_ first: Loc, _ second: Loc, _ xSize: Int) -> Int {
        let dx = getX(second, xSize) - getX(first, xSize)
        let dy = (second - first - dx) / (xSize + 1)
        return dx * dx + dy * dy
    }

    public static func toStringMach(_ loc: Loc, _ xSize: Int) -> String {
        if loc == Board.passLoc { return "pass" }
        if loc == Board.nullLoc { return "null" }
        return "(\(getX(loc, xSize)),\(getY(loc, xSize)))"
    }

    public static func toString(_ loc: Loc, _ xSize: Int, _ ySize: Int) -> String {
        if loc == Board.passLoc { return "pass" }
        if loc == Board.nullLoc { return "null" }
        let x = getX(loc, xSize)
        let y = getY(loc, xSize)
        guard x >= 0, x < xSize, y >= 0, y < ySize else { return toStringMach(loc, xSize) }
        let letters = Array("ABCDEFGHJKLMNOPQRSTUVWXYZ")
        if x <= 24 {
            return "\(letters[x])\(ySize - y)"
        }
        return "\(letters[x / 25 - 1])\(letters[x % 25])\(ySize - y)"
    }

    public static func ofString(_ input: String, _ xSize: Int, _ ySize: Int) -> Loc? {
        var trimmed = input[...]
        while trimmed.first?.isWhitespace == true {
            trimmed = trimmed.dropFirst()
        }
        while trimmed.last?.isWhitespace == true {
            trimmed = trimmed.dropLast()
        }
        let string = String(trimmed)
        if string.lowercased() == "pass" || string.lowercased() == "pss" { return Board.passLoc }
        if string.lowercased() == "null" { return Board.nullLoc }
        if string.first == "(", string.last == ")" {
            let pieces = string.dropFirst().dropLast().split(separator: ",")
            guard pieces.count == 2, let x = Int(pieces[0]), let y = Int(pieces[1]) else { return nil }
            return getLoc(x, y, xSize)
        }
        let letters = Array(string)
        guard letters.count >= 2, let firstX = letterCoordinate(letters[0]) else { return nil }
        var x = firstX
        var numberStart = 1
        if letters.count > 2, let secondX = letterCoordinate(letters[1]) {
            x = (x + 1) * 25 + secondX
            numberStart = 2
        }
        guard let row = Int(String(letters[numberStart...])), row > 0 else { return nil }
        let y = ySize - row
        guard x >= 0, x < xSize, y >= 0, y < ySize else { return nil }
        return getLoc(x, y, xSize)
    }

    private static func letterCoordinate(_ character: Character) -> Int? {
        guard let ascii = character.asciiValue else { return nil }
        let upper = ascii >= 97 ? ascii - 32 : ascii
        if upper >= 65, upper <= 72 { return Int(upper - 65) }
        if upper >= 74, upper <= 90 { return Int(upper - 66) }
        return nil
    }
}

private extension Character {
    var asciiValue: UInt8? {
        String(self).utf8.first
    }
}
