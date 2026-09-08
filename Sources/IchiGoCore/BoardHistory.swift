public final class BoardHistory: @unchecked Sendable {
    public struct EncoreKoCapture: Equatable, Sendable {
        public var posHashBeforeMove: Hash128
        public var moveLoc: Loc
        public var movePla: Player
    }

    public static let numRecentBoards = 6

    public var rules: Rules
    public private(set) var moveHistory: [Move]
    public private(set) var preventEncoreHistory: [Bool]
    public private(set) var koHashHistory: [Hash128]
    public private(set) var firstTurnIdxWithKoHistory: Int
    public private(set) var initialBoard: Board
    public private(set) var initialPla: Player
    public private(set) var initialEncorePhase: Int
    public private(set) var initialTurnNumber: Int64
    public private(set) var assumeMultipleStartingBlackMovesAreHandicap: Bool
    public private(set) var whiteHasMoved: Bool
    public private(set) var overrideNumHandicapStones: Int
    public private(set) var presumedNextMovePla: Player
    public private(set) var wasEverOccupiedOrPlayed: [Bool]
    public private(set) var superKoBanned: [Bool]
    public private(set) var consecutiveEndingPasses: Int
    public private(set) var hashesBeforeBlackPass: [Hash128]
    public private(set) var hashesBeforeWhitePass: [Hash128]
    public private(set) var encorePhase: Int
    public private(set) var numTurnsThisPhase: Int
    public private(set) var numApproxValidTurnsThisPhase: Int
    public private(set) var numConsecValidTurnsThisGame: Int
    public private(set) var koRecapBlocked: [Bool]
    public private(set) var koRecapBlockHash: Hash128
    public private(set) var koCapturesInEncore: [EncoreKoCapture]
    public private(set) var secondEncoreStartColors: [Color]
    public private(set) var whiteBonusScore: Float
    public private(set) var whiteHandicapBonusScore: Float
    public private(set) var hasButton: Bool
    public private(set) var isPastNormalPhaseEnd: Bool
    public private(set) var isGameFinished: Bool
    public private(set) var winner: Player?
    public private(set) var finalWhiteMinusBlackScore: Float
    public private(set) var isScored: Bool
    public private(set) var isNoResult: Bool
    public private(set) var isResignation: Bool

    private var recentBoards: [Board]
    private var currentRecentBoardIdx: Int

    public convenience init() {
        self.init(Board(), pla: .black, rules: Rules(), encorePhase: 0)
    }

    public init(_ board: Board, pla: Player, rules: Rules, encorePhase: Int = 0) {
        self.rules = rules
        moveHistory = []
        preventEncoreHistory = []
        koHashHistory = []
        firstTurnIdxWithKoHistory = 0
        initialBoard = board.copy()
        initialPla = pla
        initialEncorePhase = encorePhase
        initialTurnNumber = 0
        assumeMultipleStartingBlackMovesAreHandicap = false
        whiteHasMoved = false
        overrideNumHandicapStones = -1
        recentBoards = (0 ..< Self.numRecentBoards).map { _ in board.copy() }
        currentRecentBoardIdx = 0
        presumedNextMovePla = pla
        wasEverOccupiedOrPlayed = [Bool](repeating: false, count: Board.maxArrSize)
        superKoBanned = [Bool](repeating: false, count: Board.maxArrSize)
        consecutiveEndingPasses = 0
        hashesBeforeBlackPass = []
        hashesBeforeWhitePass = []
        self.encorePhase = encorePhase
        numTurnsThisPhase = 0
        numApproxValidTurnsThisPhase = 0
        numConsecValidTurnsThisGame = 0
        koRecapBlocked = [Bool](repeating: false, count: Board.maxArrSize)
        koRecapBlockHash = Hash128()
        koCapturesInEncore = []
        secondEncoreStartColors = [Color](repeating: .empty, count: Board.maxArrSize)
        whiteBonusScore = 0
        whiteHandicapBonusScore = 0
        hasButton = false
        isPastNormalPhaseEnd = false
        isGameFinished = false
        winner = nil
        finalWhiteMinusBlackScore = 0
        isScored = false
        isNoResult = false
        isResignation = false
        clear(board, pla: pla, rules: rules, encorePhase: encorePhase)
    }

    public func copy() -> BoardHistory {
        let result = BoardHistory(initialBoard, pla: initialPla, rules: rules, encorePhase: initialEncorePhase)
        result.rules = rules
        result.moveHistory = moveHistory
        result.preventEncoreHistory = preventEncoreHistory
        result.koHashHistory = koHashHistory
        result.firstTurnIdxWithKoHistory = firstTurnIdxWithKoHistory
        result.initialBoard = initialBoard.copy()
        result.initialPla = initialPla
        result.initialEncorePhase = initialEncorePhase
        result.initialTurnNumber = initialTurnNumber
        result.assumeMultipleStartingBlackMovesAreHandicap = assumeMultipleStartingBlackMovesAreHandicap
        result.whiteHasMoved = whiteHasMoved
        result.overrideNumHandicapStones = overrideNumHandicapStones
        result.recentBoards = recentBoards.map { $0.copy() }
        result.currentRecentBoardIdx = currentRecentBoardIdx
        result.presumedNextMovePla = presumedNextMovePla
        result.wasEverOccupiedOrPlayed = wasEverOccupiedOrPlayed
        result.superKoBanned = superKoBanned
        result.consecutiveEndingPasses = consecutiveEndingPasses
        result.hashesBeforeBlackPass = hashesBeforeBlackPass
        result.hashesBeforeWhitePass = hashesBeforeWhitePass
        result.encorePhase = encorePhase
        result.numTurnsThisPhase = numTurnsThisPhase
        result.numApproxValidTurnsThisPhase = numApproxValidTurnsThisPhase
        result.numConsecValidTurnsThisGame = numConsecValidTurnsThisGame
        result.koRecapBlocked = koRecapBlocked
        result.koRecapBlockHash = koRecapBlockHash
        result.koCapturesInEncore = koCapturesInEncore
        result.secondEncoreStartColors = secondEncoreStartColors
        result.whiteBonusScore = whiteBonusScore
        result.whiteHandicapBonusScore = whiteHandicapBonusScore
        result.hasButton = hasButton
        result.isPastNormalPhaseEnd = isPastNormalPhaseEnd
        result.isGameFinished = isGameFinished
        result.winner = winner
        result.finalWhiteMinusBlackScore = finalWhiteMinusBlackScore
        result.isScored = isScored
        result.isNoResult = isNoResult
        result.isResignation = isResignation
        return result
    }

    public func clear(_ board: Board, pla: Player, rules: Rules, encorePhase: Int) {
        precondition((0 ... 2).contains(encorePhase))
        precondition(encorePhase == 0 || rules.scoringRule == .territory)
        self.rules = rules
        moveHistory.removeAll()
        preventEncoreHistory.removeAll()
        koHashHistory.removeAll()
        firstTurnIdxWithKoHistory = 0
        initialBoard = board.copy()
        initialPla = pla
        initialEncorePhase = encorePhase
        initialTurnNumber = 0
        assumeMultipleStartingBlackMovesAreHandicap = false
        whiteHasMoved = false
        overrideNumHandicapStones = -1
        recentBoards = (0 ..< Self.numRecentBoards).map { _ in board.copy() }
        currentRecentBoardIdx = 0
        presumedNextMovePla = pla
        wasEverOccupiedOrPlayed = [Bool](repeating: false, count: Board.maxArrSize)
        for loc in board.playableLocations() {
            wasEverOccupiedOrPlayed[loc] = board.colors[loc] != .empty
        }
        superKoBanned = [Bool](repeating: false, count: Board.maxArrSize)
        consecutiveEndingPasses = 0
        hashesBeforeBlackPass.removeAll()
        hashesBeforeWhitePass.removeAll()
        self.encorePhase = encorePhase
        numTurnsThisPhase = 0
        numApproxValidTurnsThisPhase = 0
        numConsecValidTurnsThisGame = 0
        koRecapBlocked = [Bool](repeating: false, count: Board.maxArrSize)
        koRecapBlockHash = Hash128()
        koCapturesInEncore.removeAll()
        secondEncoreStartColors = encorePhase == 2 ? board.colors.map { $0 == .wall ? .empty : $0 } : [Color](
            repeating: .empty,
            count: Board.maxArrSize
        )
        whiteBonusScore = 0
        whiteHandicapBonusScore = 0
        hasButton = rules.hasButton && encorePhase == 0
        isPastNormalPhaseEnd = false
        isGameFinished = false
        winner = nil
        finalWhiteMinusBlackScore = 0
        isScored = false
        isNoResult = false
        isResignation = false
        koHashHistory.append(Self.getKoHash(rules, board, pla, encorePhase, koRecapBlockHash))
        if rules.scoringRule == .territory {
            for loc in board.playableLocations() {
                if board.colors[loc] == .black { whiteBonusScore += 1 }
                if board.colors[loc] == .white { whiteBonusScore -= 1 }
            }
            whiteBonusScore -= Float(board.numWhiteCaptures - board.numBlackCaptures)
        }
        whiteHandicapBonusScore = Float(computeWhiteHandicapBonus())
    }

    public func copyToInitial() -> BoardHistory {
        let history = BoardHistory(initialBoard, pla: initialPla, rules: rules, encorePhase: initialEncorePhase)
        history.setInitialTurnNumber(initialTurnNumber)
        history.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap)
        history.setOverrideNumHandicapStones(overrideNumHandicapStones)
        return history
    }

    public func setInitialTurnNumber(_ number: Int64) {
        initialTurnNumber = number
    }

    public func setAssumeMultipleStartingBlackMovesAreHandicap(_ value: Bool) {
        assumeMultipleStartingBlackMovesAreHandicap = value
        whiteHandicapBonusScore = Float(computeWhiteHandicapBonus())
    }

    public func setOverrideNumHandicapStones(_ number: Int) {
        overrideNumHandicapStones = number
        whiteHandicapBonusScore = Float(computeWhiteHandicapBonus())
    }

    public func setKomi(_ newKomi: Float) {
        let oldKomi = rules.komi
        rules.komi = newKomi
        if isGameFinished, isScored { setFinalScoreAndWinner(finalWhiteMinusBlackScore - oldKomi + newKomi) }
    }

    public func getRecentBoard(_ numMovesAgo: Int) -> Board {
        precondition((0 ..< Self.numRecentBoards).contains(numMovesAgo))
        let index = (currentRecentBoardIdx - numMovesAgo + Self.numRecentBoards) % Self.numRecentBoards
        return recentBoards[index]
    }

    public func getCurrentTurnNumber() -> Int64 {
        max(0, initialTurnNumber + Int64(moveHistory.count))
    }

    public func whiteKomiAdjustmentForDraws(_ drawEquivalentWinsForWhite: Double) -> Float {
        rules.gameResultWillBeInteger() ? Float(drawEquivalentWinsForWhite - 0.5) : 0
    }

    public func currentSelfKomi(_ pla: Player, drawEquivalentWinsForWhite: Double) -> Float {
        let whiteKomi = whiteBonusScore + whiteHandicapBonusScore + rules.komi
            + whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite)
        return pla == .white ? whiteKomi : -whiteKomi
    }

    public static func numHandicapStonesOnBoard(_ board: Board) -> Int {
        numHandicapStonesOnBoardHelper(board, blackNonPassTurnsToStart: 0)
    }

    public func computeNumHandicapStones() -> Int {
        if overrideNumHandicapStones >= 0 { return overrideNumHandicapStones }
        var blackNonPassTurnsToStart = 0
        if assumeMultipleStartingBlackMovesAreHandicap {
            for (index, move) in moveHistory.enumerated() {
                if move.pla != .black {
                    if index + 1 < moveHistory.count, moveHistory[index + 1].pla != .black {
                        blackNonPassTurnsToStart = 0
                        break
                    }
                    if move.loc == Board.passLoc { continue }
                    break
                }
                if move.loc != Board.passLoc, move.loc != Board.nullLoc { blackNonPassTurnsToStart += 1 }
            }
        }
        return Self.numHandicapStonesOnBoardHelper(initialBoard, blackNonPassTurnsToStart: blackNonPassTurnsToStart)
    }

    public func computeWhiteHandicapBonus() -> Int {
        switch rules.whiteHandicapBonusRule {
        case .zero: 0
        case .n: computeNumHandicapStones()
        case .nMinusOne: max(0, computeNumHandicapStones() - 1)
        }
    }

    public func isLegal(_ board: Board, _ moveLoc: Loc, _ movePla: Player) -> Bool {
        guard movePla == presumedNextMovePla else { return false }
        if encorePhase > 0 {
            if isPassForKo(board, moveLoc, movePla) { return true }
        } else if board.isKoBanned(moveLoc) { return false }
        return board.isLegalIgnoringKo(moveLoc, movePla, rules.multiStoneSuicideLegal) && !superKoBanned[moveLoc]
    }

    public func isPassForKo(_ board: Board, _ moveLoc: Loc, _ movePla: Player) -> Bool {
        guard encorePhase > 0, moveLoc >= 0, moveLoc < Board.maxArrSize, moveLoc != Board.passLoc else { return false }
        if board.colors[moveLoc] == movePla.opponent.color, koRecapBlocked[moveLoc],
           board.getChainSize(moveLoc) == 1, board.getNumLiberties(moveLoc) == 1 { return true }
        let captured = board.getKoCaptureLoc(moveLoc, movePla)
        return captured != Board.nullLoc && koRecapBlocked[captured] && board.colors[captured] == movePla.opponent.color
    }

    public func isLegalTolerant(_ board: Board, _ moveLoc: Loc, _ movePla: Player) -> Bool {
        isPassForKo(board, moveLoc, movePla) || board.isLegalIgnoringKo(moveLoc, movePla, true)
    }

    @discardableResult
    public func makeBoardMoveTolerant(
        _ board: Board,
        _ moveLoc: Loc,
        _ movePla: Player,
        preventEncore: Bool = false
    ) -> Bool {
        guard isLegalTolerant(board, moveLoc, movePla) else { return false }
        makeBoardMoveAssumeLegal(board, moveLoc, movePla, rootKoHashTable: nil, preventEncore: preventEncore)
        return true
    }

    public func passWouldEndPhase(_ board: Board, _ movePla: Player) -> Bool {
        let hash = Self.getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash)
        return newConsecutiveEndingPassesAfterPass() >= 2 || wouldBeSpightlikeEndingPass(movePla, hash)
    }

    public func passWouldEndGame(_ board: Board, _ movePla: Player) -> Bool {
        passWouldEndPhase(board, movePla) && isFinalPhase()
    }

    public func shouldSuppressEndGameFromFriendlyPass(_ board: Board, _ movePla: Player) -> Bool {
        rules.friendlyPassOk && rules.scoringRule == .area && newConsecutiveEndingPassesAfterPass() == 2
            && !wouldBeSpightlikeEndingPass(
                movePla,
                Self.getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash)
            )
    }

    public func isFinalPhase() -> Bool {
        rules.scoringRule == .area || (rules.scoringRule == .territory && encorePhase >= 2)
    }

    public func makeBoardMoveAssumeLegal(
        _ board: Board,
        _ moveLoc: Loc,
        _ movePla: Player,
        rootKoHashTable: KoHashTable? = nil,
        preventEncore: Bool = false
    ) {
        let posHashBeforeMove = board.posHash
        if isGameFinished || isPastNormalPhaseEnd {
            numApproxValidTurnsThisPhase = min(numApproxValidTurnsThisPhase, 1)
            numConsecValidTurnsThisGame = min(numConsecValidTurnsThisGame, 1)
        }
        let moveIsIllegal = !isLegal(board, moveLoc, movePla)
        isGameFinished = false
        isPastNormalPhaseEnd = false
        winner = nil
        finalWhiteMinusBlackScore = 0
        isScored = false
        isNoResult = false
        isResignation = false

        var isSpightlikeEndingPass = false
        if moveLoc != Board.passLoc {
            consecutiveEndingPasses = 0
        } else if hasButton {
            precondition(encorePhase == 0 && rules.hasButton)
            hasButton = false
            whiteBonusScore += movePla == .white ? 0.5 : -0.5
            consecutiveEndingPasses = 0
            hashesBeforeBlackPass.removeAll()
            hashesBeforeWhitePass.removeAll()
            koHashHistory.removeAll()
            firstTurnIdxWithKoHistory = moveHistory.count + 1
        } else {
            if phaseHasSpightlikeEndingAndPassHistoryClearing() {
                koHashHistory.removeAll()
                firstTurnIdxWithKoHistory = moveHistory.count + 1
            }
            let hashBeforeMove = Self.getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash)
            consecutiveEndingPasses = newConsecutiveEndingPassesAfterPass()
            isSpightlikeEndingPass = wouldBeSpightlikeEndingPass(movePla, hashBeforeMove)
            if movePla == .black { hashesBeforeBlackPass.append(hashBeforeMove) } else {
                hashesBeforeWhitePass.append(hashBeforeMove)
            }
        }

        var wasPassForKo = false
        if encorePhase > 0, moveLoc != Board.passLoc {
            if board.colors[moveLoc] == movePla.opponent.color, koRecapBlocked[moveLoc] {
                setKoRecapBlocked(moveLoc, false)
                wasPassForKo = true
                board.clearSimpleKoLoc()
            } else {
                let captured = board.getKoCaptureLoc(moveLoc, movePla)
                let isBlockedOpponent = captured != Board.nullLoc && koRecapBlocked[captured]
                    && board.colors[captured] == movePla.opponent.color
                if isBlockedOpponent {
                    setKoRecapBlocked(captured, false)
                    wasPassForKo = true
                    board.clearSimpleKoLoc()
                }
            }
        }
        if !wasPassForKo {
            board.playMoveAssumeLegal(moveLoc, movePla)
            if encorePhase > 0 {
                if board.koLoc != Board.nullLoc {
                    setKoRecapBlocked(moveLoc, true)
                    koCapturesInEncore.append(EncoreKoCapture(
                        posHashBeforeMove: posHashBeforeMove,
                        moveLoc: moveLoc,
                        movePla: movePla
                    ))
                    board.clearSimpleKoLoc()
                }
                for loc in board.playableLocations() where board.colors[loc] == .empty && koRecapBlocked[loc] {
                    setKoRecapBlocked(loc, false)
                }
            }
        }

        currentRecentBoardIdx = (currentRecentBoardIdx + 1) % Self.numRecentBoards
        recentBoards[currentRecentBoardIdx] = board.copy()
        koHashHistory.append(Self.getKoHash(rules, board, movePla.opponent, encorePhase, koRecapBlockHash))
        moveHistory.append(Move(loc: moveLoc, pla: movePla))
        preventEncoreHistory.append(preventEncore)
        numTurnsThisPhase += 1
        numApproxValidTurnsThisPhase += 1
        numConsecValidTurnsThisGame += 1
        presumedNextMovePla = movePla.opponent
        if moveIsIllegal { numConsecValidTurnsThisGame = 0 }
        if moveLoc != Board.passLoc { wasEverOccupiedOrPlayed[moveLoc] = true }

        let nextPla = movePla.opponent
        if encorePhase <= 0, rules.koRule != .simple {
            precondition(koRecapBlockHash == Hash128())
            for loc in board.playableLocations() {
                if board.colors[loc] != .empty {
                    superKoBanned[loc] = false
                } else if !wasEverOccupiedOrPlayed[loc], !board.isSuicide(loc, nextPla) {
                    superKoBanned[loc] = false
                } else if board.isIllegalSuicide(loc, nextPla, rules.multiStoneSuicideLegal) || loc == board.koLoc {
                    superKoBanned[loc] = false
                } else {
                    let posHash = board.getPosHashAfterMove(loc, nextPla)
                    let koHash = Self.getKoHashAfterMoveNonEncore(rules, posHash, nextPla.opponent)
                    superKoBanned[loc] = koHashOccursInHistory(koHash, rootKoHashTable)
                }
            }
        } else if encorePhase > 0 {
            superKoBanned = [Bool](repeating: false, count: Board.maxArrSize)
            let matchingCaptures = koCapturesInEncore.filter {
                $0.posHashBeforeMove == board.posHash && $0.movePla == nextPla
            }
            for capture in matchingCaptures {
                superKoBanned[capture.moveLoc] = true
            }
        }

        if rules.scoringRule == .territory, encorePhase <= 1, moveLoc != Board.passLoc, !wasPassForKo {
            whiteBonusScore += movePla == .black ? 1 : -1
        }
        if movePla == .white, moveLoc != Board.passLoc { whiteHasMoved = true }
        let shouldUpdateHandicap = assumeMultipleStartingBlackMovesAreHandicap && !whiteHasMoved
            && movePla == .black && rules.whiteHandicapBonusRule != .zero
        if shouldUpdateHandicap {
            whiteHandicapBonusScore = Float(computeWhiteHandicapBonus())
        }

        if consecutiveEndingPasses >= 2 || isSpightlikeEndingPass {
            if rules.scoringRule == .area {
                precondition(encorePhase == 0)
                endAndScoreGameNow(board)
            } else if encorePhase >= 2 {
                endAndScoreGameNow(board)
            } else if preventEncore {
                isPastNormalPhaseEnd = true
                numApproxValidTurnsThisPhase = min(numApproxValidTurnsThisPhase, 1)
                numConsecValidTurnsThisGame = min(numConsecValidTurnsThisGame, 1)
            } else {
                encorePhase += 1
                numTurnsThisPhase = 0
                numApproxValidTurnsThisPhase = 0
                if encorePhase == 2 { secondEncoreStartColors = board.colors.map { $0 == .wall ? .empty : $0 } }
                superKoBanned = [Bool](repeating: false, count: Board.maxArrSize)
                consecutiveEndingPasses = 0
                hashesBeforeBlackPass.removeAll()
                hashesBeforeWhitePass.removeAll()
                koRecapBlocked = [Bool](repeating: false, count: Board.maxArrSize)
                koRecapBlockHash = Hash128()
                koCapturesInEncore.removeAll()
                koHashHistory = [Self.getKoHash(rules, board, movePla.opponent, encorePhase, koRecapBlockHash)]
                firstTurnIdxWithKoHistory = moveHistory.count
            }
        }

        let checksLongCycle = moveLoc != Board.passLoc && (encorePhase > 0 || rules.koRule == .simple)
        if checksLongCycle, numberOfKoHashOccurrencesInHistory(koHashHistory.last!, rootKoHashTable) >= 3 {
            isNoResult = true
            isGameFinished = true
        }
    }

    public func getAreaNow(_ board: Board) -> [Color] {
        rules.scoringRule == .area ? countAreaScoreWhiteMinusBlack(board)
            .area : countTerritoryAreaScoreWhiteMinusBlack(board).area
    }

    public func endAndScoreGameNow(_ board: Board) {
        _ = endAndScoreGameNowWithArea(board)
    }

    @discardableResult
    public func endAndScoreGameNowWithArea(_ board: Board) -> [Color] {
        let result = rules
            .scoringRule == .area ? countAreaScoreWhiteMinusBlack(board) : countTerritoryAreaScoreWhiteMinusBlack(board)
        if hasButton {
            hasButton = false
            whiteBonusScore += presumedNextMovePla == .white ? 0.5 : -0.5
        }
        setFinalScoreAndWinner(Float(result.score) + whiteBonusScore + whiteHandicapBonusScore + rules.komi)
        isScored = true
        isNoResult = false
        isResignation = false
        isGameFinished = true
        isPastNormalPhaseEnd = false
        return result.area
    }

    public func endGameIfAllPassAlive(_ board: Board) {
        let area = board.calculateArea(
            nonPassAliveStones: false,
            safeBigTerritories: false,
            unsafeBigTerritories: false,
            isMultiStoneSuicideLegal: rules.multiStoneSuicideLegal
        )
        var boardScore = 0
        for loc in board.playableLocations() {
            if area[loc] == .white { boardScore += 1 } else if area[loc] == .black { boardScore -= 1 } else { return }
        }
        if rules.taxRule == .all {
            endAndScoreGameNow(board)
            return
        }
        if hasButton {
            hasButton = false
            whiteBonusScore += presumedNextMovePla == .white ? 0.5 : -0.5
        }
        setFinalScoreAndWinner(Float(boardScore) + whiteBonusScore + whiteHandicapBonusScore + rules.komi)
        isScored = true
        isNoResult = false
        isResignation = false
        isGameFinished = true
        isPastNormalPhaseEnd = false
    }

    public func setWinnerByResignation(_ pla: Player) {
        isGameFinished = true
        isPastNormalPhaseEnd = false
        isScored = false
        isNoResult = false
        isResignation = true
        winner = pla
        finalWhiteMinusBlackScore = 0
    }

    public func numberOfKoHashOccurrencesInHistory(_ hash: Hash128, _ rootKoHashTable: KoHashTable? = nil) -> Int {
        var start = 0
        var count = 0
        if let table = rootKoHashTable, firstTurnIdxWithKoHistory == table.firstTurnIdxWithKoHistory {
            count += table.numberOfOccurrencesOfHash(hash)
            start = table.size
        }
        for index in start ..< koHashHistory.count where koHashHistory[index] == hash {
            count += 1
        }
        return count
    }

    public func hasBlackPassOrWhiteFirst() -> Bool {
        if initialBoard.isEmpty(), moveHistory.first?.pla == .white { return true }
        var blackPasses = 0
        var whitePasses = 0
        var blackDoubleMoves = 0
        var whiteDoubleMoves = 0
        for index in moveHistory.indices {
            let move = moveHistory[index]
            if move.loc == Board.passLoc, move.pla == .black { blackPasses += 1 }
            if move.loc == Board.passLoc, move.pla == .white { whitePasses += 1 }
            if index > 0, move.pla == .black, moveHistory[index - 1].pla == .black { blackDoubleMoves += 1 }
            if index > 0, move.pla == .white, moveHistory[index - 1].pla == .white { whiteDoubleMoves += 1 }
        }
        return (blackPasses == 1 && whitePasses == 0 && blackDoubleMoves == 0 && whiteDoubleMoves == 0)
            || (blackPasses == 0 && whitePasses == 0 && blackDoubleMoves == 0 && whiteDoubleMoves == 1)
    }

    public func printBasicInfo(_ board: Board) -> String {
        var output = board.printBoard(history: moveHistory)
        output += "Next player: \(Self.playerString(presumedNextMovePla))\n"
        if encorePhase > 0 { output += "Game phase: \(encorePhase)\n" }
        output += "Rules: \(rules.toJsonString())\n"
        if whiteHandicapBonusScore != 0 { output += "Handicap bonus score: \(whiteHandicapBonusScore)\n" }
        output += "B stones captured: \(board.numBlackCaptures)\n"
        output += "W stones captured: \(board.numWhiteCaptures)\n"
        return output
    }

    public func printDebugInfo(_ board: Board) -> String {
        var output = "\(board)"
        output += "Initial pla \(Self.playerString(initialPla))\n"
        output += "Encore phase \(encorePhase)\n"
        output += "Turns this phase \(numTurnsThisPhase)\n"
        output += "Approx valid turns this phase \(numApproxValidTurnsThisPhase)\n"
        output += "Approx consec valid turns this game \(numConsecValidTurnsThisGame)\n"
        output += "Rules \(rules)\n"
        output += "Ko recap block hash \(koRecapBlockHash)\n"
        output += "White bonus score \(whiteBonusScore)\n"
        output += "White handicap bonus score \(whiteHandicapBonusScore)\n"
        output += "Has button \(hasButton ? 1 : 0)\n"
        output += "Presumed next pla \(Self.playerString(presumedNextMovePla))\n"
        output += "Past normal phase end \(isPastNormalPhaseEnd ? 1 : 0)\n"
        output += "Game result \(isGameFinished ? 1 : 0) \(winner.map(Self.playerString) ?? "Empty") "
        output += "\(finalWhiteMinusBlackScore) \(isScored ? 1 : 0) \(isNoResult ? 1 : 0) \(isResignation ? 1 : 0)\n"
        output += "Last moves " + moveHistory.map { Location.toString($0.loc, board.xSize, board.ySize) }
            .joined(separator: " ") + "\n"
        return output
    }

    public static func getSituationAndSimpleKoHash(_ board: Board, nextPlayer: Player) -> Hash128 {
        var hash = board.posHash ^ Board.zobrist.player[Int(nextPlayer.rawValue)]
        if board.koLoc != Board.nullLoc { hash ^= Board.zobrist.koLocation[board.koLoc] }
        return hash
    }

    public static func getSituationAndSimpleKoAndPrevPosHash(
        _ board: Board,
        history: BoardHistory,
        nextPlayer: Player
    ) -> Hash128 {
        let hash = getSituationAndSimpleKoHash(board, nextPlayer: nextPlayer)
        var mixed = Hash128(HashMix.splitMix64(hash.hash1), HashMix.rrmxmx(hash.hash0))
        if !history.moveHistory.isEmpty { mixed ^= history.getRecentBoard(1).posHash }
        return mixed
    }

    public static func getSituationRulesAndKoHash(
        _ board: Board,
        history: BoardHistory,
        nextPlayer: Player,
        drawEquivalentWinsForWhite: Double
    ) -> Hash128 {
        var hash = board.posHash ^ Board.zobrist.player[Int(nextPlayer.rawValue)]
        hash ^= Board.zobrist.encore[history.encorePhase]
        if history.encorePhase == 0 {
            if board.koLoc != Board.nullLoc { hash ^= Board.zobrist.koLocation[board.koLoc] }
            for loc in board.playableLocations() where history.superKoBanned[loc] && loc != board.koLoc {
                hash ^= Board.zobrist.koLocation[loc]
            }
        } else {
            for loc in board.playableLocations() {
                if history.superKoBanned[loc] { hash ^= Board.zobrist.koLocation[loc] }
                if history.koRecapBlocked[loc] {
                    hash ^= Board.zobrist.koMark[loc][Int(Color.black.rawValue)]
                        ^ Board.zobrist.koMark[loc][Int(Color.white.rawValue)]
                }
                if history.encorePhase == 2, history.secondEncoreStartColors[loc] != .empty {
                    hash ^= Board.zobrist.secondEncoreStart[loc][Int(history.secondEncoreStartColors[loc].rawValue)]
                }
            }
        }
        let selfKomi = history.currentSelfKomi(nextPlayer, drawEquivalentWinsForWhite: drawEquivalentWinsForWhite)
        let discretized = Int64(selfKomi * 256)
        let komiHash = HashMix.murmurMix(UInt64(bitPattern: discretized))
        hash.hash0 ^= komiHash
        hash.hash1 ^= HashMix.basicLCong(komiHash)
        hash ^= Rules.zobristKoRuleHash[history.rules.koRule.rawValue]
        hash ^= Rules.zobristScoringRuleHash[history.rules.scoringRule.rawValue]
        hash ^= Rules.zobristTaxRuleHash[history.rules.taxRule.rawValue]
        if history.rules.multiStoneSuicideLegal { hash ^= Rules.zobristMultiStoneSuicideHash }
        if history.hasButton { hash ^= Rules.zobristButtonHash }
        if history.rules.friendlyPassOk { hash ^= Rules.zobristFriendlyPassOkHash }
        return hash
    }

    private static func getKoHash(
        _ rules: Rules,
        _ board: Board,
        _ pla: Player,
        _ encorePhase: Int,
        _ koRecapBlockHash: Hash128
    ) -> Hash128 {
        if rules.koRule == .situational || rules.koRule == .simple || encorePhase > 0 {
            return board.posHash ^ Board.zobrist.player[Int(pla.rawValue)] ^ koRecapBlockHash
        }
        return board.posHash ^ koRecapBlockHash
    }

    private static func playerString(_ player: Player) -> String {
        player == .black ? "Black" : "White"
    }

    private static func getKoHashAfterMoveNonEncore(_ rules: Rules, _ posHash: Hash128, _ pla: Player) -> Hash128 {
        rules.koRule == .situational || rules.koRule == .simple
            ? posHash ^ Board.zobrist.player[Int(pla.rawValue)] : posHash
    }

    private static func numHandicapStonesOnBoardHelper(_ board: Board, blackNonPassTurnsToStart: Int) -> Int {
        let black = board.numPlaStonesOnBoard(.black)
        guard board.numPlaStonesOnBoard(.white) == 0 else { return 0 }
        let advantage = black + blackNonPassTurnsToStart
        return advantage <= 1 ? 0 : advantage
    }

    private func koHashOccursInHistory(_ hash: Hash128, _ rootKoHashTable: KoHashTable?) -> Bool {
        var start = 0
        if let table = rootKoHashTable, firstTurnIdxWithKoHistory == table.firstTurnIdxWithKoHistory {
            if table.containsHash(hash) { return true }
            start = table.size
        }
        return koHashHistory[start...].contains(hash)
    }

    private func setKoRecapBlocked(_ loc: Loc, _ value: Bool) {
        if koRecapBlocked[loc] != value {
            koRecapBlocked[loc] = value
            koRecapBlockHash ^= Board.zobrist.koMark[loc][Int(Color.black.rawValue)]
                ^ Board.zobrist.koMark[loc][Int(Color.white.rawValue)]
        }
    }

    private func countAreaScoreWhiteMinusBlack(_ board: Board) -> (score: Int, area: [Color]) {
        var score = 0
        let area: [Color]
        if rules.taxRule == .none {
            area = board.calculateArea(
                nonPassAliveStones: true,
                safeBigTerritories: true,
                unsafeBigTerritories: true,
                isMultiStoneSuicideLegal: rules.multiStoneSuicideLegal
            )
        } else {
            let result = board.calculateIndependentLifeArea(
                keepTerritories: false,
                keepStones: true,
                isMultiStoneSuicideLegal: rules.multiStoneSuicideLegal
            )
            area = result.area
            if rules.taxRule == .all { score -= 2 * result.whiteMinusBlackIndependentLifeRegionCount }
        }
        for loc in board.playableLocations() {
            if area[loc] == .white { score += 1 } else if area[loc] == .black { score -= 1 }
        }
        return (score, area)
    }

    private func countTerritoryAreaScoreWhiteMinusBlack(_ board: Board) -> (score: Int, area: [Color]) {
        let keepTerritories = rules.taxRule == .none
        let result = board.calculateIndependentLifeArea(
            keepTerritories: keepTerritories,
            keepStones: false,
            isMultiStoneSuicideLegal: rules.multiStoneSuicideLegal
        )
        var area = result.area
        var score = 0
        for loc in board.playableLocations() {
            if area[loc] == .white {
                score += 1
            } else if area[loc] == .black {
                score -= 1
            } else if board.colors[loc] == .white, encorePhase < 2 || secondEncoreStartColors[loc] == .white {
                score += 1
                area[loc] = .white
            } else if board.colors[loc] == .black, encorePhase < 2 || secondEncoreStartColors[loc] == .black {
                score -= 1
                area[loc] = .black
            }
        }
        if rules.taxRule == .all { score -= 2 * result.whiteMinusBlackIndependentLifeRegionCount }
        return (score, area)
    }

    private func setFinalScoreAndWinner(_ score: Float) {
        finalWhiteMinusBlackScore = score
        winner = score > 0 ? .white : score < 0 ? .black : nil
    }

    private func newConsecutiveEndingPassesAfterPass() -> Int {
        if encorePhase > 0 { return consecutiveEndingPasses + 1 }
        return rules.koRule == .spight ? 0 : consecutiveEndingPasses + 1
    }

    private func phaseHasSpightlikeEndingAndPassHistoryClearing() -> Bool {
        encorePhase > 0 || rules.koRule == .simple || rules.koRule == .spight
    }

    private func wouldBeSpightlikeEndingPass(_ pla: Player, _ hash: Hash128) -> Bool {
        guard phaseHasSpightlikeEndingAndPassHistoryClearing() else { return false }
        return pla == .black ? hashesBeforeBlackPass.contains(hash) : hashesBeforeWhitePass.contains(hash)
    }
}

public final class KoHashTable: @unchecked Sendable {
    public private(set) var firstTurnIdxWithKoHistory = 0
    private var occurrences: [Hash128: Int] = [:]

    public init() {}

    public var size: Int {
        occurrences.values.reduce(0, +)
    }

    public func recompute(_ history: BoardHistory) {
        occurrences.removeAll(keepingCapacity: true)
        for hash in history.koHashHistory {
            occurrences[hash, default: 0] += 1
        }
        firstTurnIdxWithKoHistory = history.firstTurnIdxWithKoHistory
    }

    public func containsHash(_ hash: Hash128) -> Bool {
        occurrences[hash] != nil
    }

    public func numberOfOccurrencesOfHash(_ hash: Hash128) -> Int {
        occurrences[hash, default: 0]
    }
}
