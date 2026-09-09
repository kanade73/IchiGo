import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures

/// Self-play driver (T21): one model plays both sides with fixed visits, temperature 0 (max
/// visits, deterministic tie-break), no resignation. Games are capped at `maxMoves`; a capped game
/// gets result "truncated" and must not be used as a win/loss target. Root visit distributions
/// are written per move as training targets (docs/spec/02-training.md §9 format precursor).
public enum SelfPlay {
    public struct GameOutput: Sendable {
        public let sgf: String
        public let targetsJSONL: String
        public let moves: Int
        public let result: String
    }

    public static func playGame(
        slot: GTPEngine.ModelSlot, size: Int, komi: Float, visits: Int, seed: UInt64, maxMoves: Int,
        settings: SearchSettings = SearchSettings()
    ) async throws -> GameOutput {
        let game = try GameState(boardSize: size, komi: komi)
        let search = try Search(evaluator: slot.evaluator, modelHash: slot.modelHash, settings: settings, initial: game.record)
        var sgfMoves: [String] = []
        var targets: [String] = []
        var n = 0
        while !game.history.isGameFinished, n < maxMoves {
            let player = game.toMove
            let r = try await search.run(visits: visits)
            guard game.isLegal(player, r.move) else { throw SearchError(message: "illegal move from search") }
            let dist = r.candidates.map { "[\($0.index),\($0.visits)]" }.joined(separator: ",")
            targets.append("{\"turn\":\(n),\"toMove\":\"\(player == .black ? "B" : "W")\",\"move\":\"\(Coordinates.gtpString(r.move, size: size))\",\"rootVisits\":\(r.rootVisits),\"visits\":[\(dist)],\"searchExpected\":\(r.searchExpected),\"rawExpected\":\(r.rootRawExpected),\"scoreLead\":\(r.searchScoreLead),\"modelHash\":\"\(slot.modelHash)\",\"seed\":\(seed)}")
            try game.play(player, r.move)
            try await search.makeMove(r.move)
            sgfMoves.append(";\(player == .black ? "B" : "W")[\(Coordinates.sgfString(r.move, size: size))]")
            n += 1
        }
        let result: String
        if let o = game.exactWhiteOutcome {
            result = o.whiteMinusBlack == 0 ? "0" : (o.whiteMinusBlack > 0 ? "W+\(o.whiteMinusBlack)" : "B+\(-o.whiteMinusBlack)")
        } else {
            result = "truncated"
        }
        let re = result == "truncated" ? "" : "RE[\(result)]"
        let sgf = "(;GM[1]FF[4]SZ[\(size)]KM[\(komi)]RU[Chinese]PB[IchiGo]PW[IchiGo]\(re)C[model \(slot.modelHash) seed \(seed) visits \(visits)]" + sgfMoves.joined() + ")\n"
        return GameOutput(sgf: sgf, targetsJSONL: targets.joined(separator: "\n") + "\n", moves: n, result: result)
    }
}
