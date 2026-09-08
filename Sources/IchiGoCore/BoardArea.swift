extension Board {
    public func calculateArea(
        nonPassAliveStones: Bool,
        safeBigTerritories: Bool,
        unsafeBigTerritories: Bool,
        isMultiStoneSuicideLegal: Bool
    ) -> [Color] {
        var result = [Color](repeating: .empty, count: Board.maxArrSize)
        calculateAreaForPla(
            .black,
            safeBigTerritories: safeBigTerritories,
            unsafeBigTerritories: unsafeBigTerritories,
            isMultiStoneSuicideLegal: isMultiStoneSuicideLegal,
            result: &result
        )
        calculateAreaForPla(
            .white,
            safeBigTerritories: safeBigTerritories,
            unsafeBigTerritories: unsafeBigTerritories,
            isMultiStoneSuicideLegal: isMultiStoneSuicideLegal,
            result: &result
        )
        if nonPassAliveStones {
            for loc in playableLocations() where result[loc] == .empty {
                result[loc] = colors[loc]
            }
        }
        return result
    }

    public func calculateArea(
        nonPassAliveStones: Bool = false,
        safeBigTerritories: Bool = false,
        unsafeBigTerritories: Bool = false
    ) -> [Color] {
        calculateArea(
            nonPassAliveStones: nonPassAliveStones,
            safeBigTerritories: safeBigTerritories,
            unsafeBigTerritories: unsafeBigTerritories,
            isMultiStoneSuicideLegal: multiStoneSuicideLegal
        )
    }

    public func calculateIndependentLifeArea(
        keepTerritories: Bool,
        keepStones: Bool,
        isMultiStoneSuicideLegal: Bool
    ) -> (area: [Color], whiteMinusBlackIndependentLifeRegionCount: Int) {
        var basicArea = [Color](repeating: .empty, count: Board.maxArrSize)
        calculateAreaForPla(
            .black,
            safeBigTerritories: true,
            unsafeBigTerritories: true,
            isMultiStoneSuicideLegal: isMultiStoneSuicideLegal,
            result: &basicArea
        )
        calculateAreaForPla(
            .white,
            safeBigTerritories: true,
            unsafeBigTerritories: true,
            isMultiStoneSuicideLegal: isMultiStoneSuicideLegal,
            result: &basicArea
        )
        for loc in playableLocations() where basicArea[loc] == .empty {
            basicArea[loc] = colors[loc]
        }
        var result = calculateIndependentLifeAreaHelper(basicArea)
        if keepTerritories {
            for loc in playableLocations() where basicArea[loc] != .empty && basicArea[loc] != colors[loc] {
                result.area[loc] = basicArea[loc]
            }
        }
        if keepStones {
            for loc in playableLocations() where basicArea[loc] != .empty && basicArea[loc] == colors[loc] {
                result.area[loc] = basicArea[loc]
            }
        }
        return result
    }

    public func calculateIndependentLifeArea(
        keepTerritories: Bool = false,
        keepStones: Bool = false
    ) -> (area: [Color], whiteMinusBlackIndependentLifeRegionCount: Int) {
        calculateIndependentLifeArea(
            keepTerritories: keepTerritories,
            keepStones: keepStones,
            isMultiStoneSuicideLegal: multiStoneSuicideLegal
        )
    }

    private func calculateAreaForPla(
        _ pla: Player,
        safeBigTerritories: Bool,
        unsafeBigTerritories: Bool,
        isMultiStoneSuicideLegal: Bool,
        result: inout [Color]
    ) {
        let opponent = pla.opponent.color
        var regionByLoc = [Int](repeating: -1, count: Board.maxArrSize)
        var regions: [AreaRegion] = []
        var atLeastOnePla = false

        for initial in playableLocations() {
            if colors[initial] == pla.color {
                atLeastOnePla = true
                continue
            }
            guard colors[initial] == .empty, regionByLoc[initial] == -1 else { continue }

            let regionIndex = regions.count
            var points: [Loc] = []
            var queue = [initial]
            var queueIndex = 0
            regionByLoc[initial] = regionIndex
            while queueIndex < queue.count {
                let loc = queue[queueIndex]
                queueIndex += 1
                points.append(loc)
                for offset in adjOffsets.prefix(4) {
                    let adjacent = loc + offset
                    if colors[adjacent] == .empty || colors[adjacent] == opponent, regionByLoc[adjacent] == -1 {
                        regionByLoc[adjacent] = regionIndex
                        queue.append(adjacent)
                    }
                }
            }

            var vitalHeads = Set<Loc>()
            for offset in adjOffsets.prefix(4) {
                let adjacent = initial + offset
                if colors[adjacent] == pla.color { vitalHeads.insert(chainHead[adjacent]) }
            }
            for loc in points where isMultiStoneSuicideLegal || colors[loc] == .empty {
                vitalHeads = vitalHeads.filter { isAdjacent(loc, toPlayerHead: $0, pla: pla) }
            }
            var internalSpaces = 0
            for loc in points where !isAdjacentToPla(loc, pla) {
                internalSpaces = min(2, internalSpaces + 1)
            }
            regions.append(AreaRegion(
                points: points,
                vitalHeads: vitalHeads,
                internalSpacesMax2: internalSpaces,
                containsOpponent: points.contains { colors[$0] == opponent },
                bordersNonPassAlive: false
            ))
        }

        let playerHeads = playableLocations().filter { colors[$0] == pla.color && chainHead[$0] == $0 }
        var killed = Set<Loc>()
        var vitalCounts: [Loc: Int] = Dictionary(uniqueKeysWithValues: playerHeads.map { ($0, 0) })
        for region in regions {
            for head in region.vitalHeads {
                vitalCounts[head, default: 0] += 1
            }
        }

        while true {
            var killedAnything = false
            for head in playerHeads where !killed.contains(head) && vitalCounts[head, default: 0] < 2 {
                killed.insert(head)
                killedAnything = true
                for index in regions.indices where !regions[index].bordersNonPassAlive {
                    if regions[index].points.contains(where: { isAdjacent($0, toPlayerHead: head, pla: pla) }) {
                        regions[index].bordersNonPassAlive = true
                        for vitalHead in regions[index].vitalHeads {
                            vitalCounts[vitalHead, default: 0] -= 1
                        }
                    }
                }
            }
            if !killedAnything { break }
        }

        for head in playerHeads where !killed.contains(head) {
            var loc = head
            repeat {
                result[loc] = pla.color
                loc = nextInChain[loc]
            } while loc != head
        }

        for region in regions {
            let passAliveTerritory = region.internalSpacesMax2 <= 1 && !region.bordersNonPassAlive && atLeastOnePla
            let safeTerritory = safeBigTerritories && !region.containsOpponent && !region
                .bordersNonPassAlive && atLeastOnePla
            if passAliveTerritory || safeTerritory {
                for loc in region.points {
                    result[loc] = pla.color
                }
            } else if unsafeBigTerritories, !region.containsOpponent, atLeastOnePla {
                for loc in region.points where result[loc] == .empty {
                    result[loc] = pla.color
                }
            }
        }
    }

    private func calculateIndependentLifeAreaHelper(
        _ basicArea: [Color]
    ) -> (area: [Color], whiteMinusBlackIndependentLifeRegionCount: Int) {
        var isSeki = [Bool](repeating: false, count: Board.maxArrSize)
        for loc in playableLocations() where basicArea[loc] != .empty && !isSeki[loc] {
            let stoneInAtari = colors[loc] == basicArea[loc] && getNumLiberties(loc) == 1
            let touchesDame = adjOffsets.prefix(4).contains {
                colors[loc + $0] == .empty && basicArea[loc + $0] == .empty
            }
            if stoneInAtari || touchesDame {
                floodArea(from: loc, color: basicArea[loc], source: basicArea) { isSeki[$0] = true }
            }
        }

        var result = [Color](repeating: .empty, count: Board.maxArrSize)
        var count = 0
        for loc in playableLocations() where basicArea[loc] != .empty && !isSeki[loc] && result[loc] != basicArea[loc] {
            let color = basicArea[loc]
            count += color == .white ? 1 : -1
            floodArea(from: loc, color: color, source: basicArea) { result[$0] = color }
        }
        return (result, count)
    }

    private func floodArea(from initial: Loc, color: Color, source: [Color], visit: (Loc) -> Void) {
        var seen: Set<Loc> = [initial]
        var queue = [initial]
        var index = 0
        while index < queue.count {
            let loc = queue[index]
            index += 1
            visit(loc)
            for offset in adjOffsets.prefix(4) {
                let adjacent = loc + offset
                if source[adjacent] == color, seen.insert(adjacent).inserted { queue.append(adjacent) }
            }
        }
    }

    private func isAdjacent(_ loc: Loc, toPlayerHead head: Loc, pla: Player) -> Bool {
        adjOffsets.prefix(4).contains {
            let adjacent = loc + $0
            return colors[adjacent] == pla.color && chainHead[adjacent] == head
        }
    }
}

private struct AreaRegion {
    let points: [Loc]
    let vitalHeads: Set<Loc>
    let internalSpacesMax2: Int
    let containsOpponent: Bool
    var bordersNonPassAlive: Bool
}
