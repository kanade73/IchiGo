/// GraphHash — superko-safe node-identity hash for the DAG/graph search.
///
/// Ported from KataGo's `cpp/game/graphhash.{h,cpp}` (see "Graph search" in
/// `docs/design-notes.md` for how the search uses it). Two layers:
///
/// - `stateHash` folds together everything relevant to *immediate* legality: the full
///   rules/komi/superko-aware position hash (`BoardHistory.getSituationRulesAndKoHash`), whether a
///   pass would end the current phase, whether the game is already over, and the number of
///   consecutive ending passes.
/// - `graphHash` is the path-chained hash called after each move: it resets to a pure `stateHash`
///   (allowing transposition merges) whenever the move that was just played proves — via the
///   existing `Board.simpleRepetitionBoundGt` — that no cycle up to `repBound` stones/liberties in
///   size could possibly land back here; otherwise it mixes the previous graph hash in, so nodes
///   reached via a different move order do NOT merge (guards against superko-relevant short cycles).
///
/// GraphHash is internal to RinGo's search (never compared against KataGo's own graph hash values,
/// unlike the position/rules hashes it's built from), so the path-mix step reuses this codebase's
/// existing `HashMix.splitMix64`/`HashMix.rrmxmx` combinator — the same pattern already used by
/// `BoardHistory.getSituationAndSimpleKoAndPrevPosHash` — rather than porting KataGo's `nasam` bit
/// for bit.
public enum GraphHash {
    // graphhash.cpp:17-18 — the only genuinely new constants this work package adds; every other
    // primitive below is reused, not reimplemented (design doc §2.1/§2.2).
    private static let consecPassMult0: UInt64 = 2_862_933_555_777_941_757
    private static let consecPassMult1: UInt64 = 3_202_034_522_624_059_733

    /// graphhash.cpp:4-22 (`GraphHash::getStateHash`). Hash of everything relevant to immediate
    /// legality: position + rules + komi + superko/ko-recap state (via the existing
    /// `getSituationRulesAndKoHash` recipe), folded with pass-ends-phase / game-over / consecutive
    /// pass count.
    public static func stateHash(
        _ hist: BoardHistory,
        _ board: Board,
        _ nextPla: Player,
        drawEquivWinsForWhite: Double
    ) -> Hash128 {
        var hash = BoardHistory.getSituationRulesAndKoHash(
            board,
            history: hist,
            nextPlayer: nextPla,
            drawEquivalentWinsForWhite: drawEquivWinsForWhite
        )
        if hist.passWouldEndPhase(board, nextPla) { hash ^= Board.zobristPassEndsPhase }
        if hist.isGameFinished { hash ^= Board.zobristGameIsOver }
        let consecutivePasses = UInt64(hist.consecutiveEndingPasses)
        hash.hash0 = hash.hash0 &+ consecPassMult0 &* consecutivePasses
        hash.hash1 = hash.hash1 &+ consecPassMult1 &* consecutivePasses
        return hash
    }

    /// graphhash.cpp:24-39 (`GraphHash::getGraphHash`). Call after making a move: `prev` is the
    /// graph hash of the position *before* that move, `prevMoveLoc` is the location that was just
    /// played (or `Board.nullLoc` if this is the very first state along the path), and `hist`/
    /// `board` describe the position the move led to.
    ///
    /// Resets to a pure `stateHash` (transposition merges allowed) whenever `prevMoveLoc` is null or
    /// `board.simpleRepetitionBoundGt(prevMoveLoc, repBound)` proves no repBound-or-smaller cycle
    /// could land back here; otherwise mixes `prev` in, so path/order affects the result (no merge).
    public static func graphHash(
        prev: Hash128,
        hist: BoardHistory,
        board: Board,
        prevMoveLoc: Loc,
        nextPla: Player,
        repBound: Int,
        drawEquivWinsForWhite: Double
    ) -> Hash128 {
        let currentStateHash = stateHash(hist, board, nextPla, drawEquivWinsForWhite: drawEquivWinsForWhite)
        guard prevMoveLoc != Board.nullLoc, !board.simpleRepetitionBoundGt(prevMoveLoc, repBound) else {
            return currentStateHash
        }
        // graphhash.cpp:31-37: mix the previous graph hash (existing splitMix64/rrmxmx combinator —
        // see the type-level doc comment for why this doesn't need to be KataGo's exact `nasam`
        // mix), then add the current state hash in (addition, not XOR — matches upstream).
        let mixed = Hash128(HashMix.splitMix64(prev.hash1), HashMix.rrmxmx(prev.hash0))
        return Hash128(mixed.hash0 &+ currentStateHash.hash0, mixed.hash1 &+ currentStateHash.hash1)
    }

    /// graphhash.cpp:41-58 (`GraphHash::getGraphHashFromScratch`). Reconstructs the graph hash chain
    /// by replaying `hist`'s full move history from its initial position (the existing
    /// `BoardHistory.copyToInitial()` corresponds to KataGo's `copyToInitial()` call at
    /// graphhash.cpp:42). Intended to be called once per search (root bootstrap), not per playout —
    /// O(move count), negligible next to one full search's GPU time.
    ///
    /// `hist`/`board` describe the *current* live position (the two are always a pair in this file,
    /// since — unlike KataGo's `BoardHistory` — RinGo's doesn't own a live "current" board itself);
    /// they're used directly for the final chain link, while the intermediate links necessarily
    /// replay through a local reconstruction since there's no other way to recover historical board
    /// states.
    public static func fromScratch(
        hist: BoardHistory,
        board: Board,
        nextPla: Player,
        repBound: Int,
        drawEquivWinsForWhite: Double
    ) -> Hash128 {
        let workingHist = hist.copyToInitial()
        let workingBoard = workingHist.initialBoard.copy()
        var chain = Hash128()
        var prevMoveLoc = Board.nullLoc
        for (index, move) in hist.moveHistory.enumerated() {
            chain = graphHash(
                prev: chain,
                hist: workingHist,
                board: workingBoard,
                prevMoveLoc: prevMoveLoc,
                nextPla: move.pla,
                repBound: repBound,
                drawEquivWinsForWhite: drawEquivWinsForWhite
            )
            let replayed = workingHist.makeBoardMoveTolerant(
                workingBoard,
                move.loc,
                move.pla,
                preventEncore: hist.preventEncoreHistory[index]
            )
            precondition(replayed, "GraphHash.fromScratch: recorded move history must replay legally")
            prevMoveLoc = move.loc
        }
        return graphHash(
            prev: chain,
            hist: hist,
            board: board,
            prevMoveLoc: prevMoveLoc,
            nextPla: nextPla,
            repBound: repBound,
            drawEquivWinsForWhite: drawEquivWinsForWhite
        )
    }
}
