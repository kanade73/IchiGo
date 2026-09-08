import Foundation
import IchiGoCore
import XCTest

final class NNInputsV7GoldenTests: XCTestCase {
    private struct Golden: Decodable {
        let hash: String
        let nextPla: Int
        let nnLen: Int
        let numSpatial: Int
        let numGlobal: Int
        let spatialNHWC: [Float]
        let global: [Float]
        let boardXSize: Int
        let boardYSize: Int
        let rules: String
        let initialStones: [[String]]
        let moves: [[String]]
    }

    func testEmpty19() throws {
        try assertGolden("empty-19")
    }

    func testOpening6() throws {
        try assertGolden("opening-6")
    }

    func testCaptureJustMade() throws {
        try assertGolden("capture-just-made")
    }

    func testSimpleKo() throws {
        try assertGolden("simple-ko")
    }

    func testWorkingLadder() throws {
        try assertGolden("working-ladder")
    }

    func testConsecutivePasses() throws {
        try assertGolden("consecutive-passes")
    }

    func testChinese9OnNN19() throws {
        try assertGolden("chinese-9-nn19")
    }

    func testJapaneseMidgame() throws {
        try assertGolden("japanese-midgame")
    }

    func testKomiHalf() throws {
        try assertGolden("komi-0.5")
    }

    func testNNPosHelpers() {
        let loc = Location.getLoc(3, 5, 9)
        XCTAssertEqual(NNPos.xyToPos(3, 5, 19), 98)
        XCTAssertEqual(NNPos.locToPos(loc, 9, 19, 19), 98)
        XCTAssertEqual(NNPos.posToLoc(98, 9, 9, 19, 19), loc)
        XCTAssertEqual(NNPos.posToLoc(18, 9, 9, 19, 19), Board.nullLoc)
        XCTAssertEqual(NNPos.getPassPos(19, 19), 361)
        XCTAssertTrue(NNPos.isPassPos(361, 19, 19))
        XCTAssertEqual(NNPos.getPolicySize(19, 19), 362)
    }

    func testGetHashChangesForEveryMiscParameter() {
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules())
        let baseline = NNInputs.getHash(board, history, .black)
        let variants = [
            MiscNNInputParams(playoutDoublingAdvantage: 1),
            MiscNNInputParams(nnPolicyTemperature: 0.8),
            MiscNNInputParams(avoidMYTDaggerHack: true),
            MiscNNInputParams(policyOptimism: 0.4),
            MiscNNInputParams(maxHistory: 0),
        ]
        for params in variants {
            XCTAssertNotEqual(NNInputs.getHash(board, history, .black, params), baseline)
        }
    }

    func testHistorySuppressionAndMaxHistory() {
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules())
        history.makeBoardMoveAssumeLegal(board, Board.passLoc, .black)
        let baseline = NNInputs.fillRowV7(board, history, .white, nnXLen: 9, nnYLen: 9)
        XCTAssertEqual(baseline.global[0], 1)
        XCTAssertEqual(baseline.global[14], 1)

        let zeroHistory = NNInputs.fillRowV7(
            board,
            history,
            .white,
            MiscNNInputParams(maxHistory: 0),
            nnXLen: 9,
            nnYLen: 9
        )
        XCTAssertEqual(zeroHistory.global[0], 0)

        let conservative = MiscNNInputParams(conservativePassAndIsRoot: true)
        let suppressed = NNInputs.fillRowV7(board, history, .white, conservative, nnXLen: 9, nnYLen: 9)
        XCTAssertEqual(suppressed.global[0], 0)
        XCTAssertEqual(suppressed.global[14], 0)
        XCTAssertNotEqual(
            NNInputs.getHash(board, history, .white, conservative),
            NNInputs.getHash(board, history, .white)
        )
    }

    func testSecondEncoreStartColorPlanes() {
        let board = Board(9, 9)
        XCTAssertTrue(board.setStone(board.getLoc(2, 2), .black))
        XCTAssertTrue(board.setStone(board.getLoc(6, 6), .white))
        let history = BoardHistory(board, pla: .black, rules: .getSimpleTerritory(), encorePhase: 2)
        let features = NNInputs.fillRowV7(board, history, .black, nnXLen: 9, nnYLen: 9)
        XCTAssertEqual(features.spatial[(2 * 9 + 2) * 22 + 20], 1)
        XCTAssertEqual(features.spatial[(6 * 9 + 6) * 22 + 21], 1)
        XCTAssertEqual(features.global[12], 1)
        XCTAssertEqual(features.global[13], 1)
    }

    func testButtonAndPlayoutDoublingGlobals() {
        let board = Board(9, 9)
        var rules = Rules()
        rules.hasButton = true
        let history = BoardHistory(board, pla: .black, rules: rules)
        let params = MiscNNInputParams(playoutDoublingAdvantage: 1.25)
        let features = NNInputs.fillRowV7(board, history, .black, params, nnXLen: 9, nnYLen: 9)
        XCTAssertEqual(features.global[15], 1)
        XCTAssertEqual(features.global[16], 0.625)
        XCTAssertEqual(features.global[17], 1)
    }

    private func assertGolden(
        _ name: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Goldens/\(name).json")
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        let rules = try Rules.parseRules(golden.rules)
        let board = Board(golden.boardXSize, golden.boardYSize)
        for stone in golden.initialStones {
            XCTAssertTrue(board.setStone(sgfLoc(stone[1], board), stone[0] == "B" ? .black : .white))
        }
        let history = BoardHistory(board, pla: .black, rules: rules)
        var nextPlayer = Player.black
        for move in golden.moves {
            let player: Player = move[0] == "B" ? .black : .white
            XCTAssertTrue(
                history.makeBoardMoveTolerant(board, sgfLoc(move[1], board), player),
                "illegal golden move \(move)",
                file: file,
                line: line
            )
            nextPlayer = player.opponent
        }

        XCTAssertEqual(Int(nextPlayer.rawValue), golden.nextPla, file: file, line: line)
        XCTAssertEqual(board.posHash.description, golden.hash, file: file, line: line)
        let result = NNInputs.fillRowV7(
            board,
            history,
            nextPlayer,
            nnXLen: golden.nnLen,
            nnYLen: golden.nnLen,
            useNHWC: true
        )
        XCTAssertEqual(golden.numSpatial, NNInputs.numFeaturesSpatialV7, file: file, line: line)
        XCTAssertEqual(golden.numGlobal, NNInputs.numFeaturesGlobalV7, file: file, line: line)
        XCTAssertEqual(result.spatial.count, golden.spatialNHWC.count, file: file, line: line)
        XCTAssertEqual(result.global.count, golden.global.count, file: file, line: line)
        for index in golden.spatialNHWC.indices {
            XCTAssertEqual(
                result.spatial[index],
                golden.spatialNHWC[index],
                accuracy: 1e-6,
                "spatial index \(index), pos \(index / 22), channel \(index % 22)",
                file: file,
                line: line
            )
        }
        for index in golden.global.indices {
            XCTAssertEqual(
                result.global[index],
                golden.global[index],
                accuracy: 1e-6,
                "global channel \(index)",
                file: file,
                line: line
            )
        }
    }

    private func sgfLoc(_ coordinate: String, _ board: Board) -> Loc {
        if coordinate.isEmpty { return Board.passLoc }
        let bytes = Array(coordinate.utf8)
        precondition(bytes.count == 2)
        return board.getLoc(Int(bytes[0] - Character("a").asciiValue!), Int(bytes[1] - Character("a").asciiValue!))
    }
}

private extension Character {
    var asciiValue: UInt8? {
        String(self).utf8.first
    }
}
