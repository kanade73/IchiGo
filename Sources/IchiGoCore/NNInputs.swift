import Foundation

public enum NNPos {
    public static let maxBoardLen = Board.maxLen
    public static let maxBoardArea = maxBoardLen * maxBoardLen
    public static let maxNNPolicySize = maxBoardArea + 1
    public static let extraScoreDistributionRadius = 60
    public static let komiClipRadius: Float = 20

    public static func xyToPos(_ x: Int, _ y: Int, _ nnXLen: Int) -> Int {
        y * nnXLen + x
    }

    public static func locToPos(_ loc: Loc, _ boardXSize: Int, _ nnXLen: Int, _ nnYLen: Int) -> Int {
        if loc == Board.passLoc { return nnXLen * nnYLen }
        if loc == Board.nullLoc { return nnXLen * (nnYLen + 1) }
        return Location.getY(loc, boardXSize) * nnXLen + Location.getX(loc, boardXSize)
    }

    public static func posToLoc(
        _ pos: Int,
        _ boardXSize: Int,
        _ boardYSize: Int,
        _ nnXLen: Int,
        _ nnYLen: Int
    ) -> Loc {
        if pos == nnXLen * nnYLen { return Board.passLoc }
        let x = pos % nnXLen
        let y = pos / nnXLen
        if x < 0 || x >= boardXSize || y < 0 || y >= boardYSize { return Board.nullLoc }
        return Location.getLoc(x, y, boardXSize)
    }

    public static func getPassPos(_ nnXLen: Int, _ nnYLen: Int) -> Int {
        nnXLen * nnYLen
    }

    public static func isPassPos(_ pos: Int, _ nnXLen: Int, _ nnYLen: Int) -> Bool {
        pos == nnXLen * nnYLen
    }

    public static func getPolicySize(_ nnXLen: Int, _ nnYLen: Int) -> Int {
        nnXLen * nnYLen + 1
    }
}

public struct MiscNNInputParams: Sendable {
    public static let symmetryNotSpecified = -1
    public static let symmetryAll = -2

    public var drawEquivalentWinsForWhite: Double
    public var conservativePassAndIsRoot: Bool
    public var enablePassingHacks: Bool
    public var playoutDoublingAdvantage: Double
    public var nnPolicyTemperature: Float
    public var avoidMYTDaggerHack: Bool
    public var symmetry: Int
    public var policyOptimism: Double
    public var maxHistory: Int

    public init(
        drawEquivalentWinsForWhite: Double = 0.5,
        conservativePassAndIsRoot: Bool = false,
        enablePassingHacks: Bool = false,
        playoutDoublingAdvantage: Double = 0,
        nnPolicyTemperature: Float = 1,
        avoidMYTDaggerHack: Bool = false,
        symmetry: Int = symmetryNotSpecified,
        policyOptimism: Double = 0,
        maxHistory: Int = 1000
    ) {
        self.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite
        self.conservativePassAndIsRoot = conservativePassAndIsRoot
        self.enablePassingHacks = enablePassingHacks
        self.playoutDoublingAdvantage = playoutDoublingAdvantage
        self.nnPolicyTemperature = nnPolicyTemperature
        self.avoidMYTDaggerHack = avoidMYTDaggerHack
        self.symmetry = symmetry
        self.policyOptimism = policyOptimism
        self.maxHistory = maxHistory
    }

    public static let zobristConservativePass = Hash128(0x0C2B_96F4_B8AE_2DA9, 0x5A14_DEE2_08FE_C0ED)
    public static let zobristFriendlyPass = Hash128(0xE750_505A_66F7_C5C2, 0x7A83_139B_F632_D6C4)
    public static let zobristPassingHacks = Hash128(0x9C89_F4FD_3CE5_A92C, 0x268C_9AFF_79C6_4D00)
    public static let zobristPlayoutDoublings = Hash128(0xA5E6_114D_380B_FC1D, 0x4160_557F_1222_F4AD)
    public static let zobristNNPolicyTemperature = Hash128(0xEBCB_DFEE_C6F4_334B, 0xB85E_43EE_243B_5AD2)
    public static let zobristAvoidMYTDaggerHack = Hash128(0x612D_22EC_402C_E054, 0x0DB9_15C4_9DE5_27AE)
    public static let zobristPolicyOptimism = Hash128(0x8841_5C85_C280_1955, 0x39BD_F76B_2AAA_5EB1)
    public static let zobristZeroHistory = Hash128(0x78F0_2AFD_D1AA_4910, 0xDA78_D550_486F_E978)
}

public enum NNInputs {
    public static let numFeaturesSpatialV7 = 22
    public static let numFeaturesGlobalV7 = 19

    public static func getHash(
        _ board: Board,
        _ history: BoardHistory,
        _ nextPlayer: Player,
        _ params: MiscNNInputParams = MiscNNInputParams()
    ) -> Hash128 {
        var hash = BoardHistory.getSituationRulesAndKoHash(
            board,
            history: history,
            nextPlayer: nextPlayer,
            drawEquivalentWinsForWhite: params.drawEquivalentWinsForWhite
        )
        if history.passWouldEndPhase(board, nextPlayer) {
            hash ^= Board.zobristPassEndsPhase
            if params.conservativePassAndIsRoot { hash ^= MiscNNInputParams.zobristConservativePass }
            if history.shouldSuppressEndGameFromFriendlyPass(board, nextPlayer) {
                hash ^= MiscNNInputParams.zobristFriendlyPass
            }
            if params.enablePassingHacks { hash ^= MiscNNInputParams.zobristPassingHacks }
        }
        if history.isGameFinished || history.isPastNormalPhaseEnd { hash ^= Board.zobristGameIsOver }
        if params.maxHistory <= 0 {
            hash.hash0 &+= MiscNNInputParams.zobristZeroHistory.hash0
            hash.hash1 &+= MiscNNInputParams.zobristZeroHistory.hash1
        }
        if params.playoutDoublingAdvantage != 0 {
            let discretized = Int64(params.playoutDoublingAdvantage * 256)
            let bits = UInt64(bitPattern: discretized)
            hash.hash0 &+= HashMix.splitMix64(bits)
            hash.hash1 &+= HashMix.basicLCong(bits)
            hash ^= MiscNNInputParams.zobristPlayoutDoublings
        }
        if params.nnPolicyTemperature != 1 {
            let discretized = Int64(params.nnPolicyTemperature * 2048)
            let bits = UInt64(bitPattern: discretized)
            hash.hash0 ^= HashMix.basicLCong2(bits)
            hash.hash1 = HashMix.splitMix64(hash.hash1 &+ bits)
            hash.hash0 &+= hash.hash1
            hash ^= MiscNNInputParams.zobristNNPolicyTemperature
        }
        if params.avoidMYTDaggerHack { hash ^= MiscNNInputParams.zobristAvoidMYTDaggerHack }
        if params.policyOptimism > 0 {
            hash ^= MiscNNInputParams.zobristPolicyOptimism
            let discretized = Int64(params.policyOptimism * 1024)
            let bits = UInt64(bitPattern: discretized)
            hash.hash0 = HashMix.rrmxmx(HashMix.splitMix64(hash.hash0) &+ bits)
            hash.hash1 = HashMix.rrmxmx(hash.hash1 &+ hash.hash0 &+ bits)
        }
        return hash
    }

    public static func fillRowV7(
        _ board: Board,
        _ history: BoardHistory,
        _ nextPlayer: Player,
        _ params: MiscNNInputParams = MiscNNInputParams(),
        nnXLen: Int,
        nnYLen: Int,
        useNHWC: Bool = true,
        rowBin: inout [Float],
        rowGlobal: inout [Float]
    ) {
        precondition(nnXLen <= NNPos.maxBoardLen && nnYLen <= NNPos.maxBoardLen)
        precondition(board.xSize <= nnXLen && board.ySize <= nnYLen)
        rowBin = [Float](repeating: 0, count: numFeaturesSpatialV7 * nnXLen * nnYLen)
        rowGlobal = [Float](repeating: 0, count: numFeaturesGlobalV7)

        let player = nextPlayer
        let opponent = player.opponent
        let xSize = board.xSize
        let ySize = board.ySize
        let featureStride = useNHWC ? 1 : nnXLen * nnYLen
        let posStride = useNHWC ? numFeaturesSpatialV7 : 1

        func set(_ pos: Int, _ feature: Int, _ value: Float = 1) {
            rowBin[pos * posStride + feature * featureStride] = value
        }

        for y in 0 ..< ySize {
            for x in 0 ..< xSize {
                let pos = NNPos.xyToPos(x, y, nnXLen)
                let loc = Location.getLoc(x, y, xSize)
                set(pos, 0)
                let stone = board.colors[loc]
                if stone == player.color { set(pos, 1) } else if stone == opponent.color { set(pos, 2) }
                if stone == player.color || stone == opponent.color {
                    switch board.getNumLiberties(loc) {
                    case 1: set(pos, 3)
                    case 2: set(pos, 4)
                    case 3: set(pos, 5)
                    default: break
                    }
                }
            }
        }

        if history.encorePhase == 0 {
            if board.koLoc != Board.nullLoc { set(NNPos.locToPos(board.koLoc, xSize, nnXLen, nnYLen), 6) }
            for loc in board.playableLocations() where history.superKoBanned[loc] && loc != board.koLoc {
                set(NNPos.locToPos(loc, xSize, nnXLen, nnYLen), 6)
            }
        } else {
            for loc in board.playableLocations() {
                let pos = NNPos.locToPos(loc, xSize, nnXLen, nnYLen)
                if history.superKoBanned[loc] { set(pos, 6) }
                if history.koRecapBlocked[loc] { set(pos, 7) }
            }
        }

        var area = [Color](repeating: .empty, count: Board.maxArrSize)
        var hasAreaFeature = false
        var groupTaxAdjustmentForPlayer = 0
        if history.rules.scoringRule == .area && history.rules.taxRule == .none {
            hasAreaFeature = true
            area = board.calculateArea(
                nonPassAliveStones: true,
                safeBigTerritories: true,
                unsafeBigTerritories: true,
                isMultiStoneSuicideLegal: history.rules.multiStoneSuicideLegal
            )
        } else {
            var keepTerritories = false
            var keepStones = false
            if history.rules.scoringRule == .area {
                hasAreaFeature = true
                keepStones = true
            } else if history.encorePhase >= 2 {
                hasAreaFeature = true
                keepTerritories = history.rules.taxRule == .none
            }
            if hasAreaFeature {
                let result = board.calculateIndependentLifeArea(
                    keepTerritories: keepTerritories,
                    keepStones: keepStones,
                    isMultiStoneSuicideLegal: history.rules.multiStoneSuicideLegal
                )
                area = result.area
                if history.rules.taxRule == .all {
                    let count = result.whiteMinusBlackIndependentLifeRegionCount
                    groupTaxAdjustmentForPlayer = player == .white ? -2 * count : 2 * count
                }
            }
        }

        var finalPhaseAndGameEndWouldNotBeWin = false
        if hasAreaFeature {
            var boardScoreForPlayer = groupTaxAdjustmentForPlayer
            for loc in board.playableLocations() {
                let pos = NNPos.locToPos(loc, xSize, nnXLen, nnYLen)
                if area[loc] == player.color {
                    set(pos, 18)
                    boardScoreForPlayer += 1
                } else if area[loc] == opponent.color {
                    set(pos, 19)
                    boardScoreForPlayer -= 1
                } else if history.rules.scoringRule == .territory {
                    if board.colors[loc] == player.color, history.secondEncoreStartColors[loc] == player.color {
                        set(pos, 18)
                        boardScoreForPlayer += 1
                    } else {
                        let isOpponentSecondEncoreStone = board.colors[loc] == opponent.color
                            && history.secondEncoreStartColors[loc] == opponent.color
                        if isOpponentSecondEncoreStone {
                            set(pos, 19)
                            boardScoreForPlayer -= 1
                        }
                    }
                }
            }
            let selfKomi = history.currentSelfKomi(
                player,
                drawEquivalentWinsForWhite: params.drawEquivalentWinsForWhite
            )
            finalPhaseAndGameEndWouldNotBeWin = Float(boardScoreForPlayer) + selfKomi <= 0
        }

        var maxTurnsOfHistoryToInclude = 5
        var suppressPassWouldEndPhase = false
        let shouldSuppressHistory = params.conservativePassAndIsRoot
            || history.shouldSuppressEndGameFromFriendlyPass(board, nextPlayer)
            || (params.enablePassingHacks && finalPhaseAndGameEndWouldNotBeWin)
        if history.passWouldEndGame(board, nextPlayer) && shouldSuppressHistory {
            maxTurnsOfHistoryToInclude = 0
            suppressPassWouldEndPhase = true
        } else if history.isGameFinished || history.isPastNormalPhaseEnd {
            maxTurnsOfHistoryToInclude = 1
        }
        maxTurnsOfHistoryToInclude = min(maxTurnsOfHistoryToInclude, params.maxHistory)

        var numTurnsOfHistoryIncluded = 0
        if maxTurnsOfHistoryToInclude > 0 {
            let moves = history.moveHistory
            let amount = min(maxTurnsOfHistoryToInclude, history.numApproxValidTurnsThisPhase)
            if amount > 0 {
                for previous in 1 ... min(5, amount) {
                    let expected = previous % 2 == 1 ? opponent : player
                    let move = moves[moves.count - previous]
                    if move.pla != expected { break }
                    numTurnsOfHistoryIncluded = previous
                    if move.loc == Board.passLoc {
                        rowGlobal[previous - 1] = 1
                    } else if move.loc != Board.nullLoc {
                        set(NNPos.locToPos(move.loc, xSize, nnXLen, nnYLen), 8 + previous)
                    }
                }
            }
        }

        iterLadders(board, nnXLen: nnXLen) { loc, pos, workingMoves in
            set(pos, 14)
            if board.colors[loc] == opponent.color, board.getNumLiberties(loc) > 1 {
                for move in workingMoves {
                    set(NNPos.locToPos(move, xSize, nnXLen, nnYLen), 17)
                }
            }
        }
        let previousBoard = numTurnsOfHistoryIncluded < 1 ? board : history.getRecentBoard(1)
        iterLadders(previousBoard, nnXLen: nnXLen) { _, pos, _ in set(pos, 15) }
        let previousPreviousBoard = numTurnsOfHistoryIncluded < 2 ? previousBoard : history.getRecentBoard(2)
        iterLadders(previousPreviousBoard, nnXLen: nnXLen) { _, pos, _ in set(pos, 16) }

        if history.encorePhase >= 2 {
            for loc in board.playableLocations() {
                let pos = NNPos.locToPos(loc, xSize, nnXLen, nnYLen)
                if history.secondEncoreStartColors[loc] == player.color { set(pos, 20) } else if history
                    .secondEncoreStartColors[loc] == opponent.color { set(pos, 21) }
            }
        }

        var selfKomi = history.currentSelfKomi(
            nextPlayer,
            drawEquivalentWinsForWhite: params.drawEquivalentWinsForWhite
        )
        let boardArea = Float(xSize * ySize)
        selfKomi = min(boardArea + NNPos.komiClipRadius, max(-boardArea - NNPos.komiClipRadius, selfKomi))
        rowGlobal[5] = selfKomi / 20

        switch history.rules.koRule {
        case .simple: break
        case .positional, .spight:
            rowGlobal[6] = 1
            rowGlobal[7] = 0.5
        case .situational:
            rowGlobal[6] = 1
            rowGlobal[7] = -0.5
        }
        if history.rules.multiStoneSuicideLegal { rowGlobal[8] = 1 }
        if history.rules.scoringRule == .territory { rowGlobal[9] = 1 }
        switch history.rules.taxRule {
        case .none: break
        case .seki: rowGlobal[10] = 1
        case .all:
            rowGlobal[10] = 1
            rowGlobal[11] = 1
        }
        if history.encorePhase > 0 { rowGlobal[12] = 1 }
        if history.encorePhase > 1 { rowGlobal[13] = 1 }
        rowGlobal[14] = suppressPassWouldEndPhase ? 0 : (history.passWouldEndPhase(board, nextPlayer) ? 1 : 0)
        if params.playoutDoublingAdvantage != 0 {
            rowGlobal[15] = 1
            rowGlobal[16] = Float(0.5 * params.playoutDoublingAdvantage)
        }
        if history.hasButton { rowGlobal[17] = 1 }

        if history.rules.scoringRule == .area || history.encorePhase >= 2 {
            let drawableKomisAreEven = (xSize * ySize) % 2 == 0
            let komiFloor: Float = if drawableKomisAreEven {
                (selfKomi / 2).rounded(.down) * 2
            } else {
                ((selfKomi - 1) / 2).rounded(.down) * 2 + 1
            }
            let delta = min(2, max(0, selfKomi - komiFloor))
            if delta < 0.5 { rowGlobal[18] = delta } else if delta < 1.5 {
                rowGlobal[18] = 1 - delta
            } else { rowGlobal[18] = delta - 2 }
        }
    }

    public static func fillRowV7(
        _ board: Board,
        _ history: BoardHistory,
        _ nextPlayer: Player,
        _ params: MiscNNInputParams = MiscNNInputParams(),
        nnXLen: Int,
        nnYLen: Int,
        useNHWC: Bool = true
    ) -> (spatial: [Float], global: [Float]) {
        var spatial: [Float] = []
        var global: [Float] = []
        fillRowV7(
            board,
            history,
            nextPlayer,
            params,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            useNHWC: useNHWC,
            rowBin: &spatial,
            rowGlobal: &global
        )
        return (spatial, global)
    }

    public static func fillScoring(_ board: Board, area: [Color], groupTax: Bool) -> [Float] {
        precondition(area.count >= Board.maxArrSize)
        var scoring = [Float](repeating: 0, count: Board.maxArrSize)
        if !groupTax {
            for loc in board.playableLocations() {
                if area[loc] == .black { scoring[loc] = -1 } else if area[loc] == .white { scoring[loc] = 1 }
            }
            return scoring
        }

        var visited = [Bool](repeating: false, count: Board.maxArrSize)
        for initial in board.playableLocations() where !visited[initial] {
            let areaColor = area[initial]
            guard areaColor == .black || areaColor == .white else { continue }
            var queue = [initial]
            var queueHead = 0
            visited[initial] = true
            var territoryCount = 0
            while queueHead < queue.count {
                let next = queue[queueHead]
                queueHead += 1
                if board.colors[next] != areaColor { territoryCount += 1 }
                for offset in board.adjOffsets.prefix(4) {
                    let adjacent = next + offset
                    if area[adjacent] == areaColor, !visited[adjacent] {
                        queue.append(adjacent)
                        visited[adjacent] = true
                    }
                }
            }
            let fullValue: Float = areaColor == .white ? 1 : -1
            let territoryValue: Float = territoryCount <= 2
                ? 0 : fullValue * Float(territoryCount - 2) / Float(territoryCount)
            for loc in queue {
                scoring[loc] = board.colors[loc] == areaColor ? fullValue : territoryValue
            }
        }
        return scoring
    }

    private static func iterLadders(
        _ board: Board,
        nnXLen: Int,
        body: (Loc, Int, [Loc]) -> Void
    ) {
        let copy = board.copy()
        var solved: [Loc: Bool] = [:]
        var buffer: [Loc] = []
        var workingMoves: [Loc] = []
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                let loc = board.getLoc(x, y)
                guard board.colors[loc].player != nil else { continue }
                let liberties = board.getNumLiberties(loc)
                guard liberties == 1 || liberties == 2 else { continue }
                let pos = NNPos.xyToPos(x, y, nnXLen)
                let head = board.chainHead[loc]
                if let laddered = solved[head] {
                    if laddered { body(loc, pos, []) }
                    continue
                }
                let laddered: Bool
                if liberties == 1 {
                    laddered = copy.searchIsLadderCaptured(loc, defenderFirst: true, buffer: &buffer)
                    workingMoves.removeAll(keepingCapacity: true)
                } else {
                    workingMoves.removeAll(keepingCapacity: true)
                    laddered = copy.searchIsLadderCapturedAttackerFirst2Libs(
                        loc,
                        buffer: &buffer,
                        workingMoves: &workingMoves
                    )
                }
                solved[head] = laddered
                if laddered { body(loc, pos, workingMoves) }
            }
        }
    }
}
