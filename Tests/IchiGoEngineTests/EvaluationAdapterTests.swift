import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import LogicModel
import XCTest

/// Fake backend (Tests only): returns fixed logits so the perspective maths can be checked exactly.
struct FakeBackend: LogicBackend {
    let name = "fake"
    var wdlLogits: [Float] = [0, 0, 0]
    var score: Float = 3
    var ownershipValue: Float = 0.25
    func evaluate(features: FeatureBatch) async throws -> RawBatch {
        let P = features.boardSize * features.boardSize
        var policy = [Float](repeating: 0, count: features.batch * (P + 1))
        for b in 0 ..< features.batch { policy[b * (P + 1) + 0] = 2 }
        return RawBatch(boardSize: features.boardSize, batch: features.batch,
                        policyLogits: policy, wdlLogits: Array(repeating: wdlLogits, count: features.batch).flatMap { $0 },
                        scoreMean: [Float](repeating: score, count: features.batch),
                        ownership: [Float](repeating: ownershipValue, count: features.batch * P))
    }
}

final class EvaluationAdapterTests: XCTestCase {
    private func eval(expected e: Float, draw: Float = 0, score: Float = 3, own: Float = 0.25, P: Int = 81) -> LogicEvaluation {
        LogicEvaluation(policy: [Float](repeating: 1 / Float(P + 1), count: P + 1), winDrawLoss: [e - draw / 2, draw, 1 - e - draw / 2],
                        expectedResult: e, scoreMean: score, ownership: [Float](repeating: own, count: P))
    }

    func testPerspectiveMaths() {
        let legal = [UInt8](repeating: 1, count: 82)
        let b = EvaluationAdapter.toWhite(eval(expected: 0.8), toMove: .black, legal: legal)
        XCTAssertEqual(b.whiteExpected, 0.2, accuracy: 1e-6)
        XCTAssertEqual(b.whiteWinValue, -0.6, accuracy: 1e-6)
        XCTAssertEqual(b.whiteScoreMean, -3); XCTAssertEqual(b.whiteLead, -3); XCTAssertEqual(b.whiteScoreMeanSq, 9)
        XCTAssertEqual(b.whiteOwnership[0], -0.25)
        XCTAssertEqual(b.whiteWinProb, 0.2, accuracy: 1e-6); XCTAssertEqual(b.whiteLossProb, 0.8, accuracy: 1e-6); XCTAssertEqual(b.whiteNoResultProb, 0)
        let w = EvaluationAdapter.toWhite(eval(expected: 0.8), toMove: .white, legal: legal)
        XCTAssertEqual(w.whiteExpected, 0.8, accuracy: 1e-6)
        XCTAssertEqual(w.whiteScoreMean, 3); XCTAssertEqual(w.whiteOwnership[0], 0.25)
        for tm in [Player.black, .white] {
            let d = EvaluationAdapter.toWhite(eval(expected: 0.5, draw: 1), toMove: tm, legal: legal)
            XCTAssertEqual(d.whiteExpected, 0.5, accuracy: 1e-6)
            XCTAssertEqual(d.whiteWinValue, 0, accuracy: 1e-6)
            XCTAssertEqual(d.rawWinDrawLoss[1], 1)
        }
    }

    func testIllegalSentinelAndLegalUnchanged() {
        var legal = [UInt8](repeating: 1, count: 82)
        legal[5] = 0; legal[81] = 0
        let w = EvaluationAdapter.toWhite(eval(expected: 0.5), toMove: .black, legal: legal)
        XCTAssertEqual(w.policy[5], -1); XCTAssertEqual(w.policy[81], -1)
        XCTAssertEqual(w.policy[0], 1 / 82, accuracy: 1e-7)
    }

    func testEvaluatorEndToEndWithFakeBackend() async throws {
        let caps = ModelCapabilities(boardSizes: [9], rulesID: IchiGoRules.rulesID, hasOwnership: true)
        let ev = LogicEvaluator(capabilities: caps, modelHash: "fake", backend: FakeBackend(wdlLogits: [log(0.6), log(0.2), log(0.2)]))
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .point(x: 0, y: 0))
        let s = g.snapshot()
        let out = try await ev.evaluate([s])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0].expectedResult, 0.7, accuracy: 1e-5)
        XCTAssertEqual(out[0].policy[0], 0)
        XCTAssertEqual(out[0].policy.reduce(0, +), 1, accuracy: 1e-5)
        let w = EvaluationAdapter.toWhite(out[0], snapshot: s)
        XCTAssertEqual(w.toMove, .white)
        XCTAssertEqual(w.whiteExpected, 0.7, accuracy: 1e-5)
        XCTAssertEqual(w.policy[0], -1)
        let g19 = try GameState(boardSize: 19, komi: 7.5)
        do { _ = try await ev.evaluate([g19.snapshot()]); XCTFail() } catch let e as EvaluatorError { XCTAssertEqual(e, .unsupportedBoardSize(19)) }
        let fin = try GameState(boardSize: 9, komi: 7)
        try fin.play(.black, .pass); try fin.play(.white, .pass)
        do { _ = try await ev.evaluate([fin.snapshot()]); XCTFail() } catch let e as EvaluatorError { XCTAssertEqual(e, .gameFinished(index: 0)) }
        let empty = try await ev.evaluate([])
        XCTAssertTrue(empty.isEmpty)
        try await ev.preWarm(size: 9)
    }

    func testEvaluatorWithRealModel() async throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/parity/tiny-9/model.ichigo")
        let model = try ModelLoader.load(directory: dir)
        let ev = LogicEvaluator(model: model, backend: ScalarBackend(model: model))
        let g = try GameState(boardSize: 9, komi: 7)
        try g.play(.black, .point(x: 4, y: 4))
        let s = g.snapshot()
        let out = try await ev.evaluate([s, s])
        XCTAssertEqual(out[0], out[1])
        XCTAssertEqual(out[0].policy[4 * 9 + 4], 0)
        XCTAssertEqual(out[0].policy.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(out[0].winDrawLoss.reduce(0, +), 1, accuracy: 1e-5)
        let w = EvaluationAdapter.toWhite(out[0], snapshot: s)
        XCTAssertEqual(w.whiteExpected, out[0].expectedResult, accuracy: 1e-6)  // white to move
    }
}
