import Foundation
import LogicModel
import XCTest

/// `make parity-cpu`: Python hard model → Swift scalar backend, every layer bit-exact,
/// heads within docs/spec/05-validation.md §3 tolerance, post-processing within 1e-5.
final class ParityTests: XCTestCase {
    func testTiny9() throws { try runCase("tiny-9") }
    func testTiny19() throws { try runCase("tiny-19") }
    /// headVersion 1 model (global head = concat(m, v, global)) must keep loading and matching.
    func testTiny9HeadV1() throws {
        let c = try ParityCase.load("tiny-9-headv1")
        XCTAssertEqual(c.model.manifest.headVersion, 1)
        try runCase("tiny-9-headv1")
    }

    private func runCase(_ name: String) throws {
        let c = try ParityCase.load(name)
        let backend = ScalarBackend(model: c.model)
        let layers = backend.layerOutputs(features: c.features)
        XCTAssertEqual(layers.count, c.model.layers)
        let perLayer = c.batch * c.boardSize * c.boardSize * c.model.channels
        for l in 0 ..< layers.count {
            let expected = Array(c.layers[(l * perLayer) ..< ((l + 1) * perLayer)])
            if layers[l] != expected {
                let firstBad = zip(layers[l], expected).enumerated().first { $0.element.0 != $0.element.1 }!.offset
                XCTFail("\(name): layer \(l) differs, first mismatch at flat index \(firstBad)")
                return
            }
        }
        let raw = try backend.evaluateSync(features: c.features)
        assertHeadClose(raw.policyLogits, c.floats("policyLogits"), "policyLogits")
        assertHeadClose(raw.wdlLogits, c.floats("wdlLogits"), "wdlLogits")
        assertHeadClose(raw.scoreMean, c.floats("scoreMean"), "scoreMean")
        assertHeadClose(raw.ownership, c.floats("ownership"), "ownership")
        let post = try Postprocess.evaluate(raw: raw, features: c.features)
        let P = c.boardSize * c.boardSize
        let expPolicy = c.floats("policy")
        let expWdl = c.floats("wdl")
        let expE = c.floats("expectedResult")
        for b in 0 ..< c.batch {
            var sum: Float = 0
            for i in 0 ..< (P + 1) {
                XCTAssertEqual(post[b].policy[i], expPolicy[b * (P + 1) + i], accuracy: 1e-5)
                if c.features.legal[b * (P + 1) + i] == 0 { XCTAssertEqual(post[b].policy[i], 0) }
                sum += post[b].policy[i]
            }
            XCTAssertEqual(sum, 1, accuracy: 1e-5)
            for i in 0 ..< 3 { XCTAssertEqual(post[b].winDrawLoss[i], expWdl[b * 3 + i], accuracy: 1e-5) }
            XCTAssertEqual(post[b].expectedResult, expE[b], accuracy: 1e-5)
        }
    }

    func testEmptyBatchReturnsEmpty() throws {
        let c = try ParityCase.load("tiny-9")
        let f = try FeatureBatch(boardSize: 9, batch: 0, spatial: [], global: [], legal: [])
        let raw = try ScalarBackend(model: c.model).evaluateSync(features: f)
        XCTAssertEqual(raw.batch, 0)
        XCTAssertTrue(raw.policyLogits.isEmpty)
    }

    func testUnsupportedBoardSizeRejected() throws {
        let c = try ParityCase.load("tiny-9")  // model declares boardSizes [9]
        let f = try FeatureBatch(boardSize: 19, batch: 1, spatial: [UInt8](repeating: 0, count: 361 * 32), global: [0, 1, 0, 0], legal: [UInt8](repeating: 1, count: 362))
        XCTAssertThrowsError(try ScalarBackend(model: c.model).evaluateSync(features: f))
    }

    func testPostprocessRejectsNoLegalMove() throws {
        let c = try ParityCase.load("tiny-9")
        let f = try FeatureBatch(boardSize: 9, batch: 1, spatial: Array(c.features.spatial[0 ..< 81 * 32]), global: Array(c.features.global[0 ..< 4]), legal: [UInt8](repeating: 0, count: 82))
        let raw = try ScalarBackend(model: c.model).evaluateSync(features: f)
        XCTAssertThrowsError(try Postprocess.evaluate(raw: raw, features: f))
    }
}
