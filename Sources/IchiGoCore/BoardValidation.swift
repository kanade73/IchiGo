public enum BoardError: Error, Equatable {
    case invalidBoardText(String)
    case inconsistentState(String)
}

public extension Board {
    static func parseBoard(
        _ xSize: Int,
        _ ySize: Int,
        _ text: String,
        lineDelimiter: Character = "\n"
    ) throws -> Board {
        let board = Board(xSize, ySize)
        var lines = text.split(separator: lineDelimiter, omittingEmptySubsequences: true).map(String.init)
        if lines.count == ySize + 1, lines[0].trimmingWhitespace().hasPrefix("A") { lines.removeFirst() }
        guard lines.count == ySize else { throw BoardError.invalidBoardText("row count differs from ySize") }
        for y in 0 ..< ySize {
            let compact = lines[y].filter { !$0.isWhitespace }
            let row = compact.drop(while: { $0.isNumber })
            guard row.count == xSize else { throw BoardError.invalidBoardText("row width differs from xSize") }
            for (x, character) in row.enumerated() {
                let loc = board.getLoc(x, y)
                switch character {
                case ".", "*", ",", "`":
                    break
                case "x", "X":
                    guard board.setStoneFailIfNoLibs(loc, .black) else {
                        throw BoardError.invalidBoardText("zero-liberty black group at \(x),\(y)")
                    }
                case "o", "O":
                    guard board.setStoneFailIfNoLibs(loc, .white) else {
                        throw BoardError.invalidBoardText("zero-liberty white group at \(x),\(y)")
                    }
                default:
                    throw BoardError.invalidBoardText("unrecognized board character \(character)")
                }
            }
        }
        return board
    }

    func checkConsistency() throws {
        var expectedHash = Board.zobrist.sizeX[xSize] ^ Board.zobrist.sizeY[ySize]
        var checked = Set<Loc>()
        for loc in 0 ..< Board.maxArrSize {
            let x = Location.getX(loc, xSize)
            let y = Location.getY(loc, xSize)
            let playable = x >= 0 && x < xSize && y >= 0 && y < ySize
            if !playable {
                guard colors[loc] == .wall else { throw BoardError.inconsistentState("non-wall outside board") }
                continue
            }
            guard colors[loc] != .wall else { throw BoardError.inconsistentState("wall inside board") }
            guard let player = colors[loc].player else { continue }
            expectedHash ^= Board.zobrist.board[loc][Int(player.color.rawValue)]
            guard !checked.contains(loc) else { continue }
            let head = chainHead[loc]
            guard head >= 0, head < Board.maxArrSize, colors[head] == player.color else {
                throw BoardError.inconsistentState("invalid chain head")
            }
            var stones: [Loc] = []
            var liberties = Set<Loc>()
            var current = loc
            repeat {
                guard current >= 0, current < Board.maxArrSize, !checked.contains(current) else {
                    throw BoardError.inconsistentState("broken circular chain")
                }
                guard colors[current] == player.color, chainHead[current] == head else {
                    throw BoardError.inconsistentState("chain color or head mismatch")
                }
                checked.insert(current)
                stones.append(current)
                for offset in adjOffsets.prefix(4) where colors[current + offset] == .empty {
                    liberties.insert(current + offset)
                }
                current = nextInChain[current]
            } while current != loc
            let data = chainData[head]
            guard data.owner == player.color, data.numLocs == stones.count,
                  data.numLiberties == liberties.count, data.numLiberties > 0
            else {
                throw BoardError.inconsistentState("chain metadata mismatch")
            }
        }
        guard expectedHash == posHash else { throw BoardError.inconsistentState("position hash mismatch") }
        if koLoc != Board.nullLoc {
            guard isOnBoard(koLoc), colors[koLoc] == .empty, getNumImmediateLiberties(koLoc) == 0 else {
                throw BoardError.inconsistentState("invalid simple ko point")
            }
        }
        guard adjOffsets == Location.getAdjacentOffsets(xSize) else {
            throw BoardError.inconsistentState("adjacency offsets mismatch")
        }
    }
}

private extension String {
    func trimmingWhitespace() -> String {
        var value = self[...]
        while value.first?.isWhitespace == true {
            value = value.dropFirst()
        }
        while value.last?.isWhitespace == true {
            value = value.dropLast()
        }
        return String(value)
    }
}
