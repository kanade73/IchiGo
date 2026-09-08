public final class Board: CustomStringConvertible, @unchecked Sendable {
    public static let maxLen = 19
    public static let defaultLen = min(maxLen, 19)
    public static let maxPlaySize = maxLen * maxLen
    public static let maxArrSize = (maxLen + 1) * (maxLen + 2) + 1
    public static let nullLoc: Loc = 0
    public static let passLoc: Loc = 1
    public static let zobristPassEndsPhase = Hash128(0x853E_097C_279E_BF4E, 0xE315_3DEF_9E14_A62C)
    public static let zobristGameIsOver = Hash128(0xB6F9_E465_597A_77EE, 0xF1D5_83D9_60A4_CE7F)

    static let zobrist = ZobristTables()

    public struct ChainData: Equatable, Sendable {
        public var owner: Color = .empty
        public var numLocs = 0
        public var numLiberties = 0
    }

    public struct MoveRecord: Sendable {
        fileprivate let snapshot: Snapshot
    }

    public let xSize: Int
    public let ySize: Int
    public private(set) var colors: [Color]
    public private(set) var chainData: [ChainData]
    public private(set) var chainHead: [Loc]
    public private(set) var nextInChain: [Loc]
    public private(set) var koLoc: Loc
    public private(set) var posHash: Hash128
    public private(set) var numBlackCaptures: Int
    public private(set) var numWhiteCaptures: Int
    public let adjOffsets: [Int]

    /// Convenience policy used by overloads that omit the C++ method's explicit suicide argument.
    public var multiStoneSuicideLegal: Bool

    /// Minimal BoardHistory-compatible support requested by this work package.
    public private(set) var wasEverOccupiedOrPlayed: [Bool]
    public private(set) var secondEncoreStartColors: [Color]

    public init(_ xSize: Int = Board.defaultLen, _ ySize: Int = Board.defaultLen) {
        precondition(xSize >= 0 && ySize >= 0 && xSize <= Board.maxLen && ySize <= Board.maxLen)
        self.xSize = xSize
        self.ySize = ySize
        colors = [Color](repeating: .wall, count: Board.maxArrSize)
        chainData = [ChainData](repeating: ChainData(), count: Board.maxArrSize)
        chainHead = [Loc](repeating: Board.nullLoc, count: Board.maxArrSize)
        nextInChain = [Loc](repeating: Board.nullLoc, count: Board.maxArrSize)
        koLoc = Board.nullLoc
        posHash = Board.zobrist.sizeX[xSize] ^ Board.zobrist.sizeY[ySize]
        numBlackCaptures = 0
        numWhiteCaptures = 0
        adjOffsets = Location.getAdjacentOffsets(xSize)
        multiStoneSuicideLegal = false
        wasEverOccupiedOrPlayed = [Bool](repeating: false, count: Board.maxArrSize)
        secondEncoreStartColors = [Color](repeating: .empty, count: Board.maxArrSize)
        for y in 0 ..< ySize {
            for x in 0 ..< xSize {
                colors[Location.getLoc(x, y, xSize)] = .empty
            }
        }
    }

    private init(snapshot: Snapshot) {
        xSize = snapshot.xSize
        ySize = snapshot.ySize
        colors = snapshot.colors
        chainData = snapshot.chainData
        chainHead = snapshot.chainHead
        nextInChain = snapshot.nextInChain
        koLoc = snapshot.koLoc
        posHash = snapshot.posHash
        numBlackCaptures = snapshot.numBlackCaptures
        numWhiteCaptures = snapshot.numWhiteCaptures
        adjOffsets = snapshot.adjOffsets
        multiStoneSuicideLegal = snapshot.multiStoneSuicideLegal
        wasEverOccupiedOrPlayed = snapshot.wasEverOccupiedOrPlayed
        secondEncoreStartColors = snapshot.secondEncoreStartColors
    }

    public static func initHash() {
        _ = zobrist
    }

    public func copy() -> Board {
        Board(snapshot: snapshot())
    }

    public func getLoc(_ x: Int, _ y: Int) -> Loc {
        Location.getLoc(x, y, xSize)
    }

    public func sqrtBoardArea() -> Double {
        xSize == ySize ? Double(xSize) : Double(xSize * ySize).squareRoot()
    }

    public func getChainSize(_ loc: Loc) -> Int {
        chainData[chainHead[loc]].numLocs
    }

    public func getNumLiberties(_ loc: Loc) -> Int {
        chainData[chainHead[loc]].numLiberties
    }

    public func getNumImmediateLiberties(_ loc: Loc) -> Int {
        adjOffsets.prefix(4).reduce(0) { $0 + (colors[loc + $1] == .empty ? 1 : 0) }
    }

    public func isOnBoard(_ loc: Loc) -> Bool {
        loc >= 0 && loc < Board.maxArrSize && colors[loc] != .wall
    }

    public func isKoBanned(_ loc: Loc) -> Bool {
        loc == koLoc
    }

    public func isSuicide(_ loc: Loc, _ pla: Player) -> Bool {
        if loc == Board.passLoc { return false }
        for offset in adjOffsets.prefix(4) {
            let adjacent = loc + offset
            if colors[adjacent] == .empty { return false }
            if colors[adjacent] == pla.color, getNumLiberties(adjacent) > 1 { return false }
            if colors[adjacent] == pla.opponent.color, getNumLiberties(adjacent) == 1 { return false }
        }
        return true
    }

    public func isIllegalSuicide(_ loc: Loc, _ pla: Player, _ isMultiStoneSuicideLegal: Bool) -> Bool {
        for offset in adjOffsets.prefix(4) {
            let adjacent = loc + offset
            if colors[adjacent] == .empty { return false }
            let friendlyChainSurvives = isMultiStoneSuicideLegal || getNumLiberties(adjacent) > 1
            if colors[adjacent] == pla.color, friendlyChainSurvives {
                return false
            }
            if colors[adjacent] == pla.opponent.color, getNumLiberties(adjacent) == 1 { return false }
        }
        return true
    }

    public func isLegal(_ loc: Loc, _ pla: Player, _ isMultiStoneSuicideLegal: Bool) -> Bool {
        loc == Board.passLoc || (isOnBoard(loc) && colors[loc] == .empty && !isKoBanned(loc)
            && !isIllegalSuicide(loc, pla, isMultiStoneSuicideLegal))
    }

    public func isLegal(_ loc: Loc, _ pla: Player) -> Bool {
        isLegal(loc, pla, multiStoneSuicideLegal)
    }

    public func isLegalIgnoringKo(_ loc: Loc, _ pla: Player, _ isMultiStoneSuicideLegal: Bool) -> Bool {
        loc == Board.passLoc || (isOnBoard(loc) && colors[loc] == .empty
            && !isIllegalSuicide(loc, pla, isMultiStoneSuicideLegal))
    }

    public func isSimpleEye(_ loc: Loc, _ pla: Player) -> Bool {
        guard colors[loc] == .empty else { return false }
        var againstWall = false
        for offset in adjOffsets.prefix(4) {
            let color = colors[loc + offset]
            if color == .wall { againstWall = true } else if color != pla.color { return false }
        }
        let opponentCorners = adjOffsets.suffix(4).reduce(0) {
            $0 + (colors[loc + $1] == pla.opponent.color ? 1 : 0)
        }
        return opponentCorners < 2 && (!againstWall || opponentCorners < 1)
    }

    public func wouldBeCapture(_ loc: Loc, _ pla: Player) -> Bool {
        guard isOnBoard(loc), colors[loc] == .empty else { return false }
        return adjOffsets.prefix(4).contains {
            colors[loc + $0] == pla.opponent.color && getNumLiberties(loc + $0) == 1
        }
    }

    public func wouldBeKoCapture(_ loc: Loc, _ pla: Player) -> Bool {
        getKoCaptureLoc(loc, pla) != Board.nullLoc
    }

    public func getKoCaptureLoc(_ loc: Loc, _ pla: Player) -> Loc {
        guard isOnBoard(loc), colors[loc] == .empty else { return Board.nullLoc }
        var capturable = Board.nullLoc
        for offset in adjOffsets.prefix(4) {
            let adjacent = loc + offset
            let color = colors[adjacent]
            if color != .wall, color != pla.opponent.color { return Board.nullLoc }
            if color == pla.opponent.color, getNumLiberties(adjacent) == 1 {
                if capturable != Board.nullLoc { return Board.nullLoc }
                capturable = adjacent
            }
        }
        guard capturable != Board.nullLoc, getChainSize(capturable) == 1 else { return Board.nullLoc }
        return capturable
    }

    public func isAdjacentToPla(_ loc: Loc, _ pla: Player) -> Bool {
        adjOffsets.prefix(4).contains { colors[loc + $0] == pla.color }
    }

    public func isAdjacentOrDiagonalToPla(_ loc: Loc, _ pla: Player) -> Bool {
        adjOffsets.contains { colors[loc + $0] == pla.color }
    }

    public func isAdjacentToChain(_ loc: Loc, _ chain: Loc) -> Bool {
        guard colors[chain].player != nil else { return false }
        return adjOffsets.prefix(4).contains {
            colors[loc + $0] == colors[chain] && chainHead[loc + $0] == chainHead[chain]
        }
    }

    public func isNonPassAliveSelfConnection(_ loc: Loc, _ pla: Player, _ passAliveArea: [Color]) -> Bool {
        guard colors[loc] == .empty, passAliveArea[loc] != pla.color else { return false }
        guard let firstHead = adjOffsets.prefix(4).lazy.map({ loc + $0 }).first(where: {
            colors[$0] == pla.color && passAliveArea[$0] == .empty
        }).map({ chainHead[$0] }) else { return false }
        return adjOffsets.prefix(4).contains {
            let adjacent = loc + $0
            return colors[adjacent] == pla.color && chainHead[adjacent] != firstHead
        }
    }

    public func isEmpty() -> Bool {
        boardLocations().allSatisfy { colors[$0] == .empty }
    }

    public func numStonesOnBoard() -> Int {
        boardLocations().count { colors[$0].player != nil }
    }

    public func numPlaStonesOnBoard(_ pla: Player) -> Int {
        boardLocations().count { colors[$0] == pla.color }
    }

    public func playableLocations() -> [Loc] {
        boardLocations()
    }

    public func getSitHashWithSimpleKo(_ pla: Player) -> Hash128 {
        var hash = posHash
        if koLoc != Board.nullLoc { hash ^= Board.zobrist.koLocation[koLoc] }
        hash ^= Board.zobrist.player[Int(pla.rawValue)]
        return hash
    }

    public func clearSimpleKoLoc() {
        koLoc = Board.nullLoc
    }

    public func setSimpleKoLoc(_ loc: Loc) {
        koLoc = loc
    }

    @discardableResult
    public func playMove(_ loc: Loc, _ pla: Player, _ isMultiStoneSuicideLegal: Bool) -> Bool {
        guard isLegal(loc, pla, isMultiStoneSuicideLegal) else { return false }
        playMoveAssumeLegal(loc, pla)
        return true
    }

    @discardableResult
    public func playMove(_ loc: Loc, _ pla: Player) -> Bool {
        playMove(loc, pla, multiStoneSuicideLegal)
    }

    public func playMoveAssumeLegal(_ loc: Loc, _ pla: Player) {
        wasEverOccupiedOrPlayed[loc] = true
        if loc == Board.passLoc {
            koLoc = Board.nullLoc
            return
        }

        colors[loc] = pla.color
        var processed = Set<Loc>()
        var captured = 0
        var possibleKoLoc = Board.nullLoc
        for offset in adjOffsets.prefix(4) {
            let adjacent = loc + offset
            guard colors[adjacent] == pla.opponent.color, !processed.contains(adjacent) else { continue }
            let group = rawGroup(at: adjacent)
            processed.formUnion(group.stones)
            if group.liberties.isEmpty {
                captured += group.stones.count
                possibleKoLoc = adjacent
                for stone in group.stones {
                    colors[stone] = .empty
                }
            }
        }

        let playedGroup = rawGroup(at: loc)
        if captured == 1, playedGroup.stones.count == 1, playedGroup.liberties.count == 1 {
            koLoc = possibleKoLoc
        } else {
            koLoc = Board.nullLoc
        }
        if pla == .black { numWhiteCaptures += captured } else { numBlackCaptures += captured }

        if playedGroup.liberties.isEmpty {
            let suicided = playedGroup.stones.count
            for stone in playedGroup.stones {
                colors[stone] = .empty
            }
            if pla == .black { numBlackCaptures += suicided } else { numWhiteCaptures += suicided }
            koLoc = Board.nullLoc
        }
        regenChainsFromColors()
    }

    public func playMoveRecorded(_ loc: Loc, _ pla: Player) -> MoveRecord {
        let record = MoveRecord(snapshot: snapshot())
        playMoveAssumeLegal(loc, pla)
        return record
    }

    public func undo(_ record: MoveRecord) {
        restore(record.snapshot)
    }

    public func getPosHashAfterMove(_ loc: Loc, _ pla: Player) -> Hash128 {
        let board = copy()
        board.playMoveAssumeLegal(loc, pla)
        return board.posHash
    }

    public func getBoundNumLibertiesAfterPlay(_ loc: Loc, _ pla: Player) -> (lower: Int, upper: Int) {
        var immediateLiberties = 0
        var captures = 0
        var potentialLibertiesFromCaptures = 0
        var connectionLiberties = 0
        var maximumConnectionLiberties = 0
        for offset in adjOffsets.prefix(4) {
            let adjacent = loc + offset
            if colors[adjacent] == .empty {
                immediateLiberties += 1
            } else if colors[adjacent] == pla.opponent.color, getNumLiberties(adjacent) == 1 {
                captures += 1
                potentialLibertiesFromCaptures += getChainSize(adjacent)
            } else if colors[adjacent] == pla.color {
                let liberties = getNumLiberties(adjacent) - 1
                connectionLiberties += liberties
                maximumConnectionLiberties = max(maximumConnectionLiberties, liberties)
            }
        }
        let lower = captures + max(maximumConnectionLiberties, immediateLiberties)
        let upper = immediateLiberties + potentialLibertiesFromCaptures + connectionLiberties
        return (lower, upper)
    }

    public func getNumLibertiesAfterPlay(_ loc: Loc, _ pla: Player, _ maximum: Int) -> Int {
        let board = copy()
        board.playMoveAssumeLegal(loc, pla)
        guard board.colors[loc] == pla.color else { return 0 }
        return min(maximum, board.getNumLiberties(loc))
    }

    @discardableResult
    public func setStone(_ loc: Loc, _ color: Color) -> Bool {
        guard isOnBoard(loc), color != .wall else { return false }
        if colors[loc] == color {
            koLoc = Board.nullLoc
            return true
        }
        let replacedStone = colors[loc] != .empty
        if replacedStone {
            colors[loc] = .empty
            regenChainsFromColors()
        }
        if let player = color.player {
            let canPlace = replacedStone ? !isSuicide(loc, player) : !isIllegalSuicide(loc, player, true)
            if canPlace { playMoveAssumeLegal(loc, player) }
        }
        koLoc = Board.nullLoc
        return true
    }

    @discardableResult
    public func setStoneFailIfNoLibs(_ loc: Loc, _ color: Color) -> Bool {
        guard isOnBoard(loc), color != .wall else { return false }
        let old = snapshot()
        colors[loc] = color
        regenChainsFromColors()
        if boardLocations().contains(where: { colors[$0].player != nil && getNumLiberties($0) == 0 }) {
            restore(old)
            return false
        }
        koLoc = Board.nullLoc
        return true
    }

    @discardableResult
    public func setStonesFailIfNoLibs(_ placements: [Move]) -> Bool {
        guard Set(placements.map(\.loc)).count == placements.count else { return false }
        guard placements.allSatisfy({ isOnBoard($0.loc) }) else { return false }
        let old = snapshot()
        for placement in placements {
            colors[placement.loc] = placement.pla.color
        }
        regenChainsFromColors()
        if boardLocations().contains(where: { colors[$0].player != nil && getNumLiberties($0) == 0 }) {
            restore(old)
            return false
        }
        koLoc = Board.nullLoc
        return true
    }

    @discardableResult
    public func setStonesTolerant(_ placements: [(loc: Loc, color: Color)]) -> Int {
        for placement in placements where isOnBoard(placement.loc) && placement.color != .wall {
            colors[placement.loc] = placement.color
        }
        regenChainsFromColors()
        var removed = 0
        for loc in boardLocations() where colors[loc].player != nil && getNumLiberties(loc) == 0 {
            colors[loc] = .empty
            removed += 1
        }
        if removed > 0 { regenChainsFromColors() }
        koLoc = Board.nullLoc
        return removed
    }

    @discardableResult
    public func setStonesTolerant(_ placements: [StonePlacement]) -> Int {
        setStonesTolerant(placements.map { (loc: $0.loc, color: $0.color) })
    }

    public func regenChainsFromColors() {
        chainData = [ChainData](repeating: ChainData(), count: Board.maxArrSize)
        chainHead = [Loc](repeating: Board.nullLoc, count: Board.maxArrSize)
        nextInChain = [Loc](repeating: Board.nullLoc, count: Board.maxArrSize)
        posHash = Board.zobrist.sizeX[xSize] ^ Board.zobrist.sizeY[ySize]
        var seen = Set<Loc>()
        for loc in boardLocations() {
            let color = colors[loc]
            guard color.player != nil else { continue }
            posHash ^= Board.zobrist.board[loc][Int(color.rawValue)]
            guard !seen.contains(loc) else { continue }
            let group = rawGroup(at: loc)
            seen.formUnion(group.stones)
            let head = loc
            chainData[head] = ChainData(owner: color, numLocs: group.stones.count, numLiberties: group.liberties.count)
            for (index, stone) in group.stones.enumerated() {
                chainHead[stone] = head
                nextInChain[stone] = group.stones[(index + 1) % group.stones.count]
            }
        }
    }

    public func recordSecondEncoreStartStones() {
        secondEncoreStartColors = colors.map { $0 == .wall ? .empty : $0 }
    }

    public func secondEncoreStartHash() -> Hash128 {
        var hash = Hash128()
        for loc in boardLocations() {
            hash ^= Board.zobrist.secondEncoreStart[loc][Int(secondEncoreStartColors[loc].rawValue)]
        }
        return hash
    }

    public func simpleRepetitionBoundGt(_ loc: Loc, _ bound: Int) -> Bool {
        guard loc != Board.nullLoc, loc != Board.passLoc else { return false }
        var count = colors[loc].player == nil ? 0 : getChainSize(loc)
        if colors[loc].player != nil, count + getNumLiberties(loc) > bound { return true }
        var seenEmpty = Set<Loc>()
        var starts: [Loc] = []
        if colors[loc] == .empty {
            starts = [loc]
        } else {
            var stone = loc
            repeat {
                starts += adjOffsets.prefix(4).map { stone + $0 }.filter { colors[$0] == .empty }
                stone = nextInChain[stone]
            } while stone != loc
        }
        for start in starts where !seenEmpty.contains(start) {
            var queue = [start]
            seenEmpty.insert(start)
            while let next = queue.popLast() {
                count += 1
                if count > bound { return true }
                for offset in adjOffsets.prefix(4) {
                    let adjacent = next + offset
                    if colors[adjacent] == .empty, seenEmpty.insert(adjacent).inserted { queue.append(adjacent) }
                }
            }
        }
        return false
    }

    public var description: String {
        printBoard()
    }

    public func printBoard(markLoc: Loc = Board.nullLoc, history: [Move]? = nil) -> String {
        var output = ""
        if let history { output += "MoveNum: \(history.count) " }
        output += "HASH: \(posHash)\n"
        let letters = Array("ABCDEFGHJKLMNOPQRSTUVWXYZ")
        output += "  "
        for x in 0 ..< xSize {
            output += x <= 24 ? " \(letters[x])" : "A\(letters[x - 25])"
        }
        output += "\n"
        for y in 0 ..< ySize {
            let row = String(ySize - y)
            output += String(repeating: " ", count: max(0, 2 - row.count)) + row + " "
            for x in 0 ..< xSize {
                let loc = getLoc(x, y)
                let character: Character = switch colors[loc] {
                case .black: "X"
                case .white: "O"
                case .empty: loc == markLoc ? "@" : "."
                case .wall: "#"
                }
                output.append(character)
                if x < xSize - 1 { output.append(" ") }
            }
            output.append("\n")
        }
        output.append("\n")
        return output
    }

    public func toStringSimple(lineDelimiter: Character = "\n") -> String {
        var result = ""
        for y in 0 ..< ySize {
            for x in 0 ..< xSize {
                let character: Character = switch colors[getLoc(x, y)] {
                case .black: "X"
                case .white: "O"
                case .empty: "."
                case .wall: "#"
                }
                result.append(character)
            }
            result.append(lineDelimiter)
        }
        return result
    }

    public func isEqualForTesting(_ other: Board, checkNumCaptures: Bool = true, checkSimpleKo: Bool = true) -> Bool {
        xSize == other.xSize && ySize == other.ySize && (!checkSimpleKo || koLoc == other.koLoc)
            && (!checkNumCaptures || (numBlackCaptures == other.numBlackCaptures
                    && numWhiteCaptures == other.numWhiteCaptures))
            && posHash == other.posHash && colors == other.colors
    }

    private func boardLocations() -> [Loc] {
        (0 ..< ySize).flatMap { y in (0 ..< xSize).map { x in Location.getLoc(x, y, xSize) } }
    }

    private func rawGroup(at initial: Loc) -> (stones: [Loc], liberties: Set<Loc>) {
        let color = colors[initial]
        guard color.player != nil else { return ([], []) }
        var stones: [Loc] = []
        var liberties = Set<Loc>()
        var seen: Set<Loc> = [initial]
        var queue = [initial]
        while let loc = queue.popLast() {
            stones.append(loc)
            for offset in adjOffsets.prefix(4) {
                let adjacent = loc + offset
                if colors[adjacent] == .empty {
                    liberties.insert(adjacent)
                } else if colors[adjacent] == color, seen.insert(adjacent).inserted {
                    queue.append(adjacent)
                }
            }
        }
        return (stones, liberties)
    }

    private func snapshot() -> Snapshot {
        Snapshot(
            xSize: xSize,
            ySize: ySize,
            colors: colors,
            chainData: chainData,
            chainHead: chainHead,
            nextInChain: nextInChain,
            koLoc: koLoc,
            posHash: posHash,
            numBlackCaptures: numBlackCaptures,
            numWhiteCaptures: numWhiteCaptures,
            adjOffsets: adjOffsets,
            multiStoneSuicideLegal: multiStoneSuicideLegal,
            wasEverOccupiedOrPlayed: wasEverOccupiedOrPlayed,
            secondEncoreStartColors: secondEncoreStartColors
        )
    }

    private func restore(_ snapshot: Snapshot) {
        precondition(xSize == snapshot.xSize && ySize == snapshot.ySize)
        colors = snapshot.colors
        chainData = snapshot.chainData
        chainHead = snapshot.chainHead
        nextInChain = snapshot.nextInChain
        koLoc = snapshot.koLoc
        posHash = snapshot.posHash
        numBlackCaptures = snapshot.numBlackCaptures
        numWhiteCaptures = snapshot.numWhiteCaptures
        multiStoneSuicideLegal = snapshot.multiStoneSuicideLegal
        wasEverOccupiedOrPlayed = snapshot.wasEverOccupiedOrPlayed
        secondEncoreStartColors = snapshot.secondEncoreStartColors
    }
}

private struct Snapshot {
    let xSize: Int
    let ySize: Int
    let colors: [Color]
    let chainData: [Board.ChainData]
    let chainHead: [Loc]
    let nextInChain: [Loc]
    let koLoc: Loc
    let posHash: Hash128
    let numBlackCaptures: Int
    let numWhiteCaptures: Int
    let adjOffsets: [Int]
    let multiStoneSuicideLegal: Bool
    let wasEverOccupiedOrPlayed: [Bool]
    let secondEncoreStartColors: [Color]
}
