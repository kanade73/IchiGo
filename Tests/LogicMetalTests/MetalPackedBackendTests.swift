import Foundation
import LogicMetal
import LogicModel
import XCTest

/// `make check-metal`/`make parity-metal`: `MetalPackedBackend` (docs/spec/04-tasks.md T23/T24)
/// -- packed gate layers bit-exact against `ScalarBackend`/`PackedCPUBackend`, and Metal heads
/// within docs/spec/05-validation.md §3 tolerance of the CPU heads, including on real positions.
final class MetalPackedBackendTests: XCTestCase {
    let batches = [0, 1, 2, 31, 32, 33, 63, 64, 65]

    // MARK: - Gate-layer parity (T23 acceptance: "B=1,31,32,33,63,64,65の全gate/全layer一致")

    func testGateLayerParityTiny9() async throws { try await runGateSweep("tiny-9") }
    func testGateLayerParityTiny19() async throws { try await runGateSweep("tiny-19") }
    func testGateLayerParityHeadV1() async throws { try await runGateSweep("tiny-9-headv1") }
    func testGateLayerParityHeadV3Tiny9() async throws { try await runGateSweep("tiny-9-headv3") }
    func testGateLayerParityHeadV3Tiny19() async throws { try await runGateSweep("tiny-19-headv3") }

    private func runGateSweep(_ fixture: String) async throws {
        try requireMetal()
        let c = try ParityCase.load(fixture)
        let scalar = ScalarBackend(model: c.model)
        let packedCPU = PackedCPUBackend(model: c.model)
        let metalPacked = try MetalPackedBackend(model: c.model)
        for b in batches {
            let features = try syntheticFeatures(boardSize: c.boardSize, batch: b, seed: UInt64(0x9ACC_B000 + b))
            let scalarLayers = scalar.layerOutputs(features: features)
            let cpuPackedLayers = packedCPU.layerOutputs(features: features)
            let metalLayers = try await metalPacked.layerOutputs(features: features)
            XCTAssertEqual(metalLayers.count, scalarLayers.count, "\(fixture) batch \(b)")
            for l in 0 ..< scalarLayers.count {
                XCTAssertEqual(cpuPackedLayers[l], scalarLayers[l], "\(fixture) batch \(b) layer \(l) cpu-packed vs scalar")
                XCTAssertEqual(metalLayers[l], scalarLayers[l], "\(fixture) batch \(b) layer \(l) metal-packed vs scalar")
            }
        }
    }

    // MARK: - Heads (T24 acceptance: "CPU FP32 tolerance以内、policy順位と最終prob検証")

    func testHeadsWithinToleranceTiny9() async throws { try await runHeadsCase("tiny-9") }
    func testHeadsWithinToleranceTiny19() async throws { try await runHeadsCase("tiny-19") }
    func testHeadsWithinToleranceHeadV1() async throws {
        let c = try ParityCase.load("tiny-9-headv1")
        XCTAssertEqual(c.model.manifest.headVersion, 1)
        try await runHeadsCase("tiny-9-headv1")
    }
    func testHeadsWithinToleranceHeadV3Tiny9() async throws {
        let c = try ParityCase.load("tiny-9-headv3")
        XCTAssertEqual(c.model.manifest.headVersion, 3)
        try await runHeadsCase("tiny-9-headv3")
    }
    func testHeadsWithinToleranceHeadV3Tiny19() async throws {
        let c = try ParityCase.load("tiny-19-headv3")
        XCTAssertEqual(c.model.manifest.headVersion, 3)
        try await runHeadsCase("tiny-19-headv3")
    }

    private func runHeadsCase(_ fixture: String) async throws {
        try requireMetal()
        let c = try ParityCase.load(fixture)
        let metalPacked = try MetalPackedBackend(model: c.model)

        // Against the fixed Python-golden fixture (same oracle `ParityTests`/`MetalParityTests` use).
        let raw = try await metalPacked.evaluate(features: c.features)
        assertHeadClose(raw.policyLogits, c.floats("policyLogits"), "\(fixture) policyLogits")
        assertHeadClose(raw.wdlLogits, c.floats("wdlLogits"), "\(fixture) wdlLogits")
        assertHeadClose(raw.scoreMean, c.floats("scoreMean"), "\(fixture) scoreMean")
        assertHeadClose(raw.ownership, c.floats("ownership"), "\(fixture) ownership")

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

        // Against the CPU scalar backend directly, at a batch size that straddles a group
        // boundary (33 -> two packed groups, one partial).
        let scalar = ScalarBackend(model: c.model)
        for b in [1, 2, 33] {
            let features = try syntheticFeatures(boardSize: c.boardSize, batch: b, seed: UInt64(0xCAFE_0000 + b))
            let rawScalar = try scalar.evaluateSync(features: features)
            let rawMetal = try await metalPacked.evaluate(features: features)
            assertHeadClose(rawMetal.policyLogits, rawScalar.policyLogits, "\(fixture) batch \(b) policyLogits")
            assertHeadClose(rawMetal.wdlLogits, rawScalar.wdlLogits, "\(fixture) batch \(b) wdlLogits")
            assertHeadClose(rawMetal.scoreMean, rawScalar.scoreMean, "\(fixture) batch \(b) scoreMean")
            assertHeadClose(rawMetal.ownership, rawScalar.ownership, "\(fixture) batch \(b) ownership")
        }
    }

    func testEmptyBatchReturnsEmpty() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9")
        let metalPacked = try MetalPackedBackend(model: c.model)
        let f = try FeatureBatch(boardSize: 9, batch: 0, spatial: [], global: [], legal: [])
        let raw = try await metalPacked.evaluate(features: f)
        XCTAssertEqual(raw.batch, 0)
        XCTAssertTrue(raw.policyLogits.isEmpty)
        let layers = try await metalPacked.layerOutputs(features: f)
        XCTAssertEqual(layers.count, c.model.layers)
        XCTAssertTrue(layers.allSatisfy(\.isEmpty))
    }

    func testUnsupportedBoardSizeThrows() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9") // model declares boardSizes [9]
        let metalPacked = try MetalPackedBackend(model: c.model)
        let f = try FeatureBatch(
            boardSize: 19, batch: 1, spatial: [UInt8](repeating: 0, count: 361 * 32),
            global: [0, 1, 0, 0], legal: [UInt8](repeating: 1, count: 362)
        )
        do {
            _ = try await metalPacked.evaluate(features: f)
            XCTFail("expected an error for an unsupported board size")
        } catch let e as LogicModelError {
            guard case .invalidInput = e else { return XCTFail("expected .invalidInput, got \(e)") }
        }
    }

    func testDeterministicAcrossRepeatedRuns() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9")
        let metalPacked = try MetalPackedBackend(model: c.model)
        let first = try await metalPacked.layerOutputs(features: c.features)
        let second = try await metalPacked.layerOutputs(features: c.features)
        XCTAssertEqual(first, second)
        let rawFirst = try await metalPacked.evaluate(features: c.features)
        let rawSecond = try await metalPacked.evaluate(features: c.features)
        XCTAssertEqual(rawFirst, rawSecond)
    }

    // MARK: - Real-position ranking (deliverable: "ranking check on real positions from
    // data/positions-9.jsonl (first 32)")

    /// The production `p2-small-gl10` model against the first 32 rows of `data/positions-9.jsonl`:
    /// CPU scalar heads vs. Metal-packed (packed gates + Metal heads) must pick the same top move
    /// (unless the CPU top-2 logit margin is itself a near-tie -- docs/spec/05-validation.md §3:
    /// "near-tieのargmax着手差はlogit marginも記録する", i.e. a near-tie flip is not a bug) and
    /// every post-processed probability must land within the usual 1e-5 tolerance.
    func testRealPositionRankingMatchesCPU() async throws {
        try requireMetal()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let modelDir = root.appendingPathComponent("models/p2-small-gl10.ichigo")
        let positionsPath = root.appendingPathComponent("data/positions-9.jsonl")
        guard FileManager.default.fileExists(atPath: modelDir.path), FileManager.default.fileExists(atPath: positionsPath.path) else {
            throw XCTSkip("models/p2-small-gl10.ichigo or data/positions-9.jsonl not present in this checkout")
        }
        let model = try ModelLoader.load(directory: modelDir)
        let rows = try loadPositionRows(positionsPath, count: 32, boardSize: model.manifest.boardSizes[0])
        XCTAssertEqual(rows.count, 32)

        var spatial: [UInt8] = [], global: [Float] = [], legal: [UInt8] = []
        for r in rows { spatial += r.spatial; global += r.global; legal += r.legal }
        let features = try FeatureBatch(boardSize: model.manifest.boardSizes[0], batch: rows.count, spatial: spatial, global: global, legal: legal)

        let scalar = ScalarBackend(model: model)
        let metalPacked = try MetalPackedBackend(model: model)
        let rawScalar = try scalar.evaluateSync(features: features)
        let rawMetal = try await metalPacked.evaluate(features: features)
        let postScalar = try Postprocess.evaluate(raw: rawScalar, features: features)
        let postMetal = try Postprocess.evaluate(raw: rawMetal, features: features)

        for i in 0 ..< rows.count {
            let ps = postScalar[i].policy
            let pm = postMetal[i].policy
            XCTAssertEqual(pm.count, ps.count)
            for j in 0 ..< ps.count {
                XCTAssertEqual(pm[j], ps[j], accuracy: 1e-5, "position \(i) policy[\(j)]")
            }
            let sortedDesc = ps.sorted(by: >)
            let margin = sortedDesc.count > 1 ? sortedDesc[0] - sortedDesc[1] : Float.infinity
            let idxScalar = ps.indices.max(by: { ps[$0] < ps[$1] })!
            let idxMetal = pm.indices.max(by: { pm[$0] < pm[$1] })!
            if idxScalar != idxMetal {
                XCTAssertLessThan(margin, 1e-3, "position \(i): top move differs (\(idxScalar) vs \(idxMetal)) without a near-tie margin (\(margin))")
            }
        }
    }

    private func loadPositionRows(_ path: URL, count: Int, boardSize: Int) throws -> [(spatial: [UInt8], global: [Float], legal: [UInt8])] {
        let text = try String(contentsOf: path, encoding: .utf8)
        var out: [(spatial: [UInt8], global: [Float], legal: [UInt8])] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if out.count >= count { break }
            guard let data = line.data(using: .utf8),
                  let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let size = obj["boardSize"] as? Int, size == boardSize,
                  let spatialAny = obj["spatial"] as? [NSNumber],
                  let globalAny = obj["global"] as? [NSNumber],
                  let legalAny = obj["legal"] as? [NSNumber]
            else { continue }
            let spatial = spatialAny.map { UInt8(truncating: $0) }
            let global = globalAny.map { Float(truncating: $0) }
            let legal = legalAny.map { UInt8(truncating: $0) }
            out.append((spatial, global, legal))
        }
        return out
    }
}
