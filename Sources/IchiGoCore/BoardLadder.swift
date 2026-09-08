extension Board {
    public func searchIsLadderCapturedAttackerFirst2Libs(
        _ loc: Loc,
        buffer: inout [Loc],
        workingMoves: inout [Loc]
    ) -> Bool {
        guard isOnBoard(loc), colors[loc].player != nil, getNumLiberties(loc) == 2 else { return false }
        let defender = colors[loc].player!
        let attacker = defender.opponent
        let liberties = chainLiberties(loc)
        buffer = liberties
        var successfulMoves: [Loc] = []
        for move in liberties where isLegal(move, attacker, false) {
            let record = playMoveRecorded(move, attacker)
            let works = searchIsLadderCaptured(loc, defenderFirst: true, buffer: &buffer)
            undo(record)
            if works { successfulMoves.append(move) }
        }
        if !successfulMoves.isEmpty {
            workingMoves = successfulMoves
            return true
        }
        return false
    }

    public func searchIsLadderCaptured(_ loc: Loc, defenderFirst: Bool, buffer: inout [Loc]) -> Bool {
        guard isOnBoard(loc), colors[loc].player != nil else { return false }
        let liberties = getNumLiberties(loc)
        guard liberties <= 2, !defenderFirst || liberties <= 1 else { return false }
        let oldKo = koLoc
        if defenderFirst { clearSimpleKoLoc() }
        var nodes = 0
        let result = ladderSearch(
            target: loc,
            defender: colors[loc].player!,
            defenderTurn: defenderFirst,
            depth: 0,
            nodes: &nodes
        )
        setSimpleKoLoc(oldKo)
        buffer = colors[loc].player == nil ? [] : chainLiberties(loc)
        return result
    }

    private func ladderSearch(
        target: Loc,
        defender: Player,
        defenderTurn: Bool,
        depth: Int,
        nodes: inout Int
    ) -> Bool {
        guard colors[target] == defender.color else { return true }
        let libertyCount = getNumLiberties(target)
        if !defenderTurn, libertyCount <= 1 { return true }
        if !defenderTurn, libertyCount >= 3 { return false }
        if defenderTurn, libertyCount >= 2 { return false }
        if defenderTurn, koLoc != Board.nullLoc { return false }
        if depth >= xSize * ySize * 3 / 2 { return true }
        if nodes >= 25000 { return false }

        let liberties = chainLiberties(target)
        var moves = liberties
        if defenderTurn {
            moves = libertyGainingCaptures(target)
            for liberty in liberties where !moves.contains(liberty) {
                moves.append(liberty)
            }
            let bounds = getBoundNumLibertiesAfterPlay(liberties[0], defender)
            if bounds.lower >= 3 { return false }
            if moves.count == 1, bounds.upper <= 1 { return true }
        } else {
            precondition(liberties.count == 2)
            let attacker = defender.opponent
            var firstImmediateLiberties = getNumImmediateLiberties(liberties[0])
            var secondImmediateLiberties = getNumImmediateLiberties(liberties[1])
            let isDoubleKoDeath = firstImmediateLiberties == 0
                && secondImmediateLiberties == 0
                && wouldBeKoCapture(liberties[0], attacker)
                && wouldBeKoCapture(liberties[1], attacker)
                && getNumLibertiesAfterPlay(liberties[0], defender, 3) <= 2
                && getNumLibertiesAfterPlay(liberties[1], defender, 3) <= 2
                && libertyGainingCaptures(target).isEmpty
            if isDoubleKoDeath {
                return true
            }
            if !Location.isAdjacent(liberties[0], liberties[1], xSize) {
                if firstImmediateLiberties >= 3, secondImmediateLiberties >= 3 { return false }
                if firstImmediateLiberties >= 3 {
                    moves = [liberties[0]]
                } else if secondImmediateLiberties >= 3 {
                    moves = [liberties[1]]
                }
            }
            if moves.count > 1 {
                firstImmediateLiberties = firstImmediateLiberties * 2
                    + countHeuristicConnectionLibertiesX2(liberties[0], defender)
                secondImmediateLiberties = secondImmediateLiberties * 2
                    + countHeuristicConnectionLibertiesX2(liberties[1], defender)
                if secondImmediateLiberties > firstImmediateLiberties {
                    moves.swapAt(0, 1)
                }
            }
        }
        let player = defenderTurn ? defender : defender.opponent
        for move in moves where isLegal(move, player, false) {
            let record = playMoveRecorded(move, player)
            nodes += 1
            let captured = ladderSearch(
                target: target,
                defender: defender,
                defenderTurn: !defenderTurn,
                depth: depth + 1,
                nodes: &nodes
            )
            undo(record)
            if defenderTurn, !captured { return false }
            if !defenderTurn, captured { return true }
        }
        return defenderTurn
    }

    private func chainLiberties(_ loc: Loc) -> [Loc] {
        var result: [Loc] = []
        var stone = loc
        repeat {
            for offset in adjOffsets.prefix(4) {
                let liberty = stone + offset
                if colors[liberty] == .empty, !result.contains(liberty) { result.append(liberty) }
            }
            stone = nextInChain[stone]
        } while stone != loc
        return result
    }

    private func libertyGainingCaptures(_ loc: Loc) -> [Loc] {
        guard let defender = colors[loc].player else { return [] }
        var checkedHeads = Set<Loc>()
        var result: [Loc] = []
        var stone = loc
        repeat {
            for offset in adjOffsets.prefix(4) {
                let adjacent = stone + offset
                if colors[adjacent] == defender.opponent.color {
                    let head = chainHead[adjacent]
                    if checkedHeads.insert(head).inserted, getNumLiberties(adjacent) == 1 {
                        result += chainLiberties(adjacent)
                    }
                }
            }
            stone = nextInChain[stone]
        } while stone != loc
        return result
    }

    private func countHeuristicConnectionLibertiesX2(_ loc: Loc, _ player: Player) -> Int {
        adjOffsets.prefix(4).reduce(into: 0) { count, offset in
            let adjacent = loc + offset
            if colors[adjacent] == player.color {
                let liberties = getNumLiberties(adjacent)
                if liberties > 1 { count += liberties * 2 - 3 }
            }
        }
    }
}
