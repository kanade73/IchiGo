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

    // MARK: - ValueSource (docs/spec/03-engine.md §3-4 "値ソース")

    /// Hand-computed per the task spec: black to move, Σownership=+10, komi 7 → komiSelf=-7 →
    /// score_est=3 → value_own=sigmoid((3+1)/6), in the to-move (black) perspective.
    func testOwnershipValueHandComputedBlackToMove() {
        let legal = [UInt8](repeating: 1, count: 82)
        var own = [Float](repeating: 0, count: 81)
        own[0] = 10  // Σownership == 10 exactly
        let e = LogicEvaluation(policy: [Float](repeating: 1 / 82, count: 82), winDrawLoss: [0.5, 0, 0.5], expectedResult: 0.5, scoreMean: 0, ownership: own)
        let w = EvaluationAdapter.toWhite(e, toMove: .black, legal: legal, komi: 7, valueSource: .ownership(k: 6, b: 1))
        let expectedOwnBlack: Float = 1 / (1 + exp(-(3 + 1) / 6))
        XCTAssertEqual(w.whiteExpected, 1 - expectedOwnBlack, accuracy: 1e-6)  // black to move: whiteExpected = 1 - toMoveExpected
        XCTAssertEqual(w.whiteWinValue, 2 * (1 - expectedOwnBlack) - 1, accuracy: 1e-6)
        XCTAssertEqual(w.whiteScoreMean, -3, accuracy: 1e-6)   // score lead: sign(black=-1) * score_est(3)
        XCTAssertEqual(w.whiteLead, -3, accuracy: 1e-6)
        // raw NN WDL is always kept, regardless of value source.
        XCTAssertEqual(w.rawWinDrawLoss, [0.5, 0, 0.5])
    }

    /// The same position from white's perspective: score_est's komiSelf flips sign.
    func testOwnershipValueHandComputedWhiteToMove() {
        let legal = [UInt8](repeating: 1, count: 82)
        var own = [Float](repeating: 0, count: 81)
        own[0] = 10
        let e = LogicEvaluation(policy: [Float](repeating: 1 / 82, count: 82), winDrawLoss: [0.5, 0, 0.5], expectedResult: 0.5, scoreMean: 0, ownership: own)
        let w = EvaluationAdapter.toWhite(e, toMove: .white, legal: legal, komi: 7, valueSource: .ownership(k: 6, b: 1))
        // score_est = 10 + 7 = 17 → value_own = sigmoid((17+1)/6), toMove == white so whiteExpected == toMoveExpected.
        let expectedOwnWhite: Float = 1 / (1 + exp(-(17 + 1) / 6))
        XCTAssertEqual(w.whiteExpected, expectedOwnWhite, accuracy: 1e-6)
        XCTAssertEqual(w.whiteScoreMean, 17, accuracy: 1e-6)
    }

    /// `.blend(weightNetwork: 1, ...)` must reduce exactly to `.network`'s value/win fields;
    /// `.blend(weightNetwork: 0, ...)` must reduce exactly to `.ownership`'s. Score lead always
    /// follows the ownership estimate whenever mode != .network (documented in
    /// `EvaluationAdapter.toWhite`), so it is checked against `.ownership`, not `.network`, for
    /// both blend weights.
    func testBlendWeightsReduceToPureModes() {
        let legal = [UInt8](repeating: 1, count: 82)
        for toMove: Player in [.black, .white] {
            let e = eval(expected: 0.8, score: 3, own: 0.25)
            let network = EvaluationAdapter.toWhite(e, toMove: toMove, legal: legal, komi: 7, valueSource: .network)
            let ownership = EvaluationAdapter.toWhite(e, toMove: toMove, legal: legal, komi: 7, valueSource: .ownership(k: 6, b: 1))
            let blendAllNetwork = EvaluationAdapter.toWhite(e, toMove: toMove, legal: legal, komi: 7, valueSource: .blend(weightNetwork: 1, k: 6, b: 1))
            let blendAllOwnership = EvaluationAdapter.toWhite(e, toMove: toMove, legal: legal, komi: 7, valueSource: .blend(weightNetwork: 0, k: 6, b: 1))

            XCTAssertEqual(blendAllNetwork.whiteExpected, network.whiteExpected, accuracy: 1e-6)
            XCTAssertEqual(blendAllNetwork.whiteWinValue, network.whiteWinValue, accuracy: 1e-6)
            XCTAssertEqual(blendAllOwnership.whiteExpected, ownership.whiteExpected, accuracy: 1e-6)
            XCTAssertEqual(blendAllOwnership.whiteWinValue, ownership.whiteWinValue, accuracy: 1e-6)

            // Score lead: both blend weights follow the ownership estimate, not the network's.
            XCTAssertEqual(blendAllNetwork.whiteScoreMean, ownership.whiteScoreMean, accuracy: 1e-6)
            XCTAssertEqual(blendAllOwnership.whiteScoreMean, ownership.whiteScoreMean, accuracy: 1e-6)
            XCTAssertNotEqual(network.whiteScoreMean, ownership.whiteScoreMean)  // sanity: the two sources actually differ here

            // Raw NN WDL is unaffected by mode.
            XCTAssertEqual(blendAllNetwork.rawWinDrawLoss, network.rawWinDrawLoss)
            XCTAssertEqual(blendAllOwnership.rawWinDrawLoss, network.rawWinDrawLoss)
        }
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
