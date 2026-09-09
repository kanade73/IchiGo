import Foundation
import LogicModel
import XCTest

/// `Heads.evaluateAccelerated` (docs/spec/04-tasks.md T24: "CPU Heads.evaluateを...高速化") must
/// stay within docs/spec/05-validation.md §3 tolerance of the explicit-loop `Heads.evaluate`,
/// which remains the oracle. Exercised at several batch sizes, including ones that don't divide
/// evenly, on both tiny fixtures and both headVersions.
final class HeadsAcceleratedTests: XCTestCase {
    func testMatchesExplicitLoopTiny9() throws { try runCase("tiny-9") }
    func testMatchesExplicitLoopTiny19() throws { try runCase("tiny-19") }
    func testMatchesExplicitLoopHeadV1() throws { try runCase("tiny-9-headv1") }

    private func runCase(_ fixture: String) throws {
        let c = try ParityCase.load(fixture)
        let scalar = ScalarBackend(model: c.model)
        for b in [1, 2, 3, 5, 8, 17, 32] {
            let features = try syntheticFeatures(boardSize: c.boardSize, batch: b, seed: UInt64(0xACCE_0000 + b))
            let layers = scalar.layerOutputs(features: features)
            let lastLayer = layers[c.model.layers - 1]
            let reference = try Heads.evaluate(model: c.model, lastLayer: lastLayer, features: features)
            let fast = try Heads.evaluateAccelerated(model: c.model, lastLayer: lastLayer, features: features)
            assertHeadClose(fast.policyLogits, reference.policyLogits, "\(fixture) batch \(b) policyLogits")
            assertHeadClose(fast.wdlLogits, reference.wdlLogits, "\(fixture) batch \(b) wdlLogits")
            assertHeadClose(fast.scoreMean, reference.scoreMean, "\(fixture) batch \(b) scoreMean")
            assertHeadClose(fast.ownership, reference.ownership, "\(fixture) batch \(b) ownership")

            // Post-processing on top of the accelerated heads should also land within the usual
            // policy/prob tolerances against the explicit-loop reference's post-processing.
            let postRef = try Postprocess.evaluate(raw: reference, features: features)
            let postFast = try Postprocess.evaluate(raw: fast, features: features)
            for i in 0 ..< postRef.count {
                for j in 0 ..< postRef[i].policy.count {
                    XCTAssertEqual(postFast[i].policy[j], postRef[i].policy[j], accuracy: 1e-4, "\(fixture) batch \(b) sample \(i) policy[\(j)]")
                }
            }
        }
    }

    /// `evaluateAccelerated` on `[Float]` non-finite checks match `evaluate`'s contract.
    func testStillThrowsOnEmptyIsUnreachedButRuns() throws {
        let c = try ParityCase.load("tiny-9")
        let scalar = ScalarBackend(model: c.model)
        let layers = scalar.layerOutputs(features: c.features)
        let raw = try Heads.evaluateAccelerated(model: c.model, lastLayer: layers[c.model.layers - 1], features: c.features)
        XCTAssertTrue(raw.policyLogits.allSatisfy(\.isFinite))
    }
}
