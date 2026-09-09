import Foundation
import LogicMetal
import LogicModel
import XCTest

/// `make parity-metal` (docs/spec/05-validation.md §1): CPU scalar vs Metal-byte on the same
/// fixed fixtures `parity-cpu` uses, both board sizes. Every logic layer must be bit-exact and
/// the FP32 heads must be within docs/spec/05-validation.md §3 tolerance -- checked two ways:
/// directly against `ScalarBackend` (the literal "parity vs ScalarBackend" requirement) and
/// against the Python-golden `expected.json` the fixture ships (the same oracle `ParityTests`
/// uses), so a bug that happens to make both Swift backends agree with each other but not with
/// the golden model would still be caught.
final class MetalParityTests: XCTestCase {
    func testTiny9() async throws { try await runCase("tiny-9") }
    func testTiny19() async throws { try await runCase("tiny-19") }
    /// headVersion 1 model (global head = concat(m, v, global)) must keep loading and matching.
    func testTiny9HeadV1() async throws {
        let c = try ParityCase.load("tiny-9-headv1")
        XCTAssertEqual(c.model.manifest.headVersion, 1)
        try await runCase("tiny-9-headv1")
    }
    /// headVersion 3 model (global head = concat(m, v, zbar, zreg, ownMean, global)) must load and match.
    func testTiny9HeadV3() async throws {
        let c = try ParityCase.load("tiny-9-headv3")
        XCTAssertEqual(c.model.manifest.headVersion, 3)
        try await runCase("tiny-9-headv3")
    }
    func testTiny19HeadV3() async throws {
        let c = try ParityCase.load("tiny-19-headv3")
        XCTAssertEqual(c.model.manifest.headVersion, 3)
        try await runCase("tiny-19-headv3")
    }

    private func runCase(_ name: String) async throws {
        try requireMetal()
        let c = try ParityCase.load(name)
        let metal = try MetalBackend(model: c.model)
        let scalar = ScalarBackend(model: c.model)

        let metalLayers = try await metal.layerOutputs(features: c.features)
        let scalarLayers = scalar.layerOutputs(features: c.features)
        XCTAssertEqual(metalLayers.count, c.model.layers)
        XCTAssertEqual(metalLayers.count, scalarLayers.count)

        let perLayer = c.batch * c.boardSize * c.boardSize * c.model.channels
        for l in 0 ..< metalLayers.count {
            if metalLayers[l] != scalarLayers[l] {
                let firstBad = zip(metalLayers[l], scalarLayers[l]).enumerated().first { $0.element.0 != $0.element.1 }!.offset
                XCTFail("\(name): layer \(l) differs from ScalarBackend, first mismatch at flat index \(firstBad)")
                return
            }
            let expected = Array(c.layers[(l * perLayer) ..< ((l + 1) * perLayer)])
            if metalLayers[l] != expected {
                let firstBad = zip(metalLayers[l], expected).enumerated().first { $0.element.0 != $0.element.1 }!.offset
                XCTFail("\(name): layer \(l) differs from golden fixture, first mismatch at flat index \(firstBad)")
                return
            }
        }

        let raw = try await metal.evaluate(features: c.features)
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
}
