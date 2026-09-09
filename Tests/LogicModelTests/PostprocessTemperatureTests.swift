import Foundation
import LogicModel
import XCTest

/// docs/spec/03-engine.md §9 / docs/spec/05-validation.md §5 (T29 calibration): `temperature`
/// divides the wdl logits before their softmax and never touches the policy softmax.
///
/// The golden `winDrawLoss`/`expectedResult` values below are `softmax([2,0,-2]/T)` computed
/// independently (numpy), and are mirrored exactly by
/// `Training/tests/test_wiring_model.py::test_postprocess_temperature_scales_wdl_logits_only` --
/// together these are the "Swift/Python postprocess agree on a small example" check for T29.
final class PostprocessTemperatureTests: XCTestCase {
    /// Minimal valid batch (board size 1: one point + pass) isolating the wdl/temperature math
    /// from the rest of postprocessing.
    private func batch(wdlLogits: [Float]) throws -> (RawBatch, FeatureBatch) {
        let features = try FeatureBatch(boardSize: 1, batch: 1, spatial: [UInt8](repeating: 0, count: 32),
                                        global: [Float](repeating: 0, count: 4), legal: [1, 0])
        let raw = RawBatch(boardSize: 1, batch: 1, policyLogits: [0, 0], wdlLogits: wdlLogits, scoreMean: [0], ownership: [0])
        return (raw, features)
    }

    func testTemperatureOneIsUnchangedAndMatchesTheDefault() throws {
        let (raw, features) = try batch(wdlLogits: [2, 0, -2])
        let expectedWDL: [Float] = [0.8668133, 0.1173104, 0.0158762]
        let byDefault = try Postprocess.evaluate(raw: raw, features: features)
        let explicitT1 = try Postprocess.evaluate(raw: raw, features: features, temperature: 1.0)
        for i in 0 ..< 3 {
            XCTAssertEqual(byDefault[0].winDrawLoss[i], expectedWDL[i], accuracy: 1e-6)
            XCTAssertEqual(explicitT1[0].winDrawLoss[i], expectedWDL[i], accuracy: 1e-6)
        }
        XCTAssertEqual(byDefault[0].expectedResult, 0.9254685, accuracy: 1e-6)
        XCTAssertEqual(byDefault[0].policy, explicitT1[0].policy)
    }

    func testTemperatureTwoHalvesTheLogitGap() throws {
        let (raw, features) = try batch(wdlLogits: [2, 0, -2])
        let t1 = try Postprocess.evaluate(raw: raw, features: features, temperature: 1.0)
        let t2 = try Postprocess.evaluate(raw: raw, features: features, temperature: 2.0)
        let expectedWDL: [Float] = [0.6652410, 0.2447285, 0.0900306]
        for i in 0 ..< 3 {
            XCTAssertEqual(t2[0].winDrawLoss[i], expectedWDL[i], accuracy: 1e-6)
            // T=2's logit gap (log-odds between any two outcomes) is exactly half T=1's.
            let gap1 = log(Double(t1[0].winDrawLoss[i])) - log(Double(t1[0].winDrawLoss[(i + 1) % 3]))
            let gap2 = log(Double(t2[0].winDrawLoss[i])) - log(Double(t2[0].winDrawLoss[(i + 1) % 3]))
            XCTAssertEqual(gap2, gap1 / 2, accuracy: 1e-6)
        }
        XCTAssertEqual(t2[0].expectedResult, 0.7876052, accuracy: 1e-6)
        // Temperature only scales the wdl logits; the (already trivial here) policy is untouched.
        XCTAssertEqual(t1[0].policy, t2[0].policy)
        XCTAssertEqual(t2[0].policy, [1, 0])
    }
}
