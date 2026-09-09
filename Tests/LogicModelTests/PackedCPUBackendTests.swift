import Foundation
@testable import LogicModel
import XCTest

/// A structurally valid `FeatureBatch` with pseudo-random 0/1 spatial/legal bits and bounded
/// global floats, at an arbitrary batch size (mirrors `Tests/LogicMetalTests/TestSupport.swift`'s
/// helper of the same name/shape, duplicated here since test targets cannot share `internal`
/// code).
func syntheticFeatures(boardSize: Int, batch: Int, seed: UInt64) throws -> FeatureBatch {
    var rng = SplitMix64(seed: seed)
    let S = boardSize
    var spatial = [UInt8](repeating: 0, count: batch * S * S * 32)
    for i in 0 ..< spatial.count { spatial[i] = UInt8(rng.next() & 1) }
    var global = [Float](repeating: 0, count: batch * 4)
    for i in 0 ..< global.count { global[i] = Float(Int64(rng.next() % 2001) - 1000) / 1000 }
    var legal = [UInt8](repeating: 0, count: batch * (S * S + 1))
    for i in 0 ..< legal.count { legal[i] = UInt8(rng.next() & 1) }
    return try FeatureBatch(boardSize: S, batch: batch, spatial: spatial, global: global, legal: legal)
}

/// `make check-cpu`: `PackedCPUBackend` vs `ScalarBackend`, docs/spec/04-tasks.md T23 acceptance
/// ("B=1,31,32,33,63,64,65の全gate/全layer一致、NOT/true末尾padding=0").
final class PackedCPUBackendTests: XCTestCase {
    let batches = [0, 1, 2, 31, 32, 33, 63, 64, 65]

    func testBatchSweepMatchesScalarBackendTiny9() throws { try runSweep("tiny-9") }
    func testBatchSweepMatchesScalarBackendTiny19() throws { try runSweep("tiny-19") }
    func testBatchSweepMatchesScalarBackendHeadV1() throws { try runSweep("tiny-9-headv1") }
    func testBatchSweepMatchesScalarBackendHeadV3Tiny9() throws { try runSweep("tiny-9-headv3") }
    func testBatchSweepMatchesScalarBackendHeadV3Tiny19() throws { try runSweep("tiny-19-headv3") }
    func testBatchSweepMatchesScalarBackendLUT4Tiny9() throws { try runSweep("tiny-9-lut4") }
    func testBatchSweepMatchesScalarBackendLUT4Tiny19() throws { try runSweep("tiny-19-lut4") }

    private func runSweep(_ fixture: String) throws {
        let c = try ParityCase.load(fixture)
        let scalar = ScalarBackend(model: c.model)
        let packed = PackedCPUBackend(model: c.model)
        for b in batches {
            let features = try syntheticFeatures(boardSize: c.boardSize, batch: b, seed: UInt64(0x9ACC_0000 + b))
            let scalarLayers = scalar.layerOutputs(features: features)
            let packedLayers = packed.layerOutputs(features: features)
            XCTAssertEqual(scalarLayers.count, packedLayers.count, "\(fixture) batch \(b)")
            for l in 0 ..< scalarLayers.count {
                XCTAssertEqual(scalarLayers[l], packedLayers[l], "\(fixture) batch \(b) layer \(l)")
            }
            let rawScalar = try scalar.evaluateSync(features: features)
            let rawPacked = try packed.evaluateSync(features: features)
            XCTAssertEqual(rawPacked.batch, b)
            assertHeadClose(rawPacked.policyLogits, rawScalar.policyLogits, "\(fixture) batch \(b) policyLogits")
            assertHeadClose(rawPacked.wdlLogits, rawScalar.wdlLogits, "\(fixture) batch \(b) wdlLogits")
            assertHeadClose(rawPacked.scoreMean, rawScalar.scoreMean, "\(fixture) batch \(b) scoreMean")
            assertHeadClose(rawPacked.ownership, rawScalar.ownership, "\(fixture) batch \(b) ownership")
        }
    }

    func testEmptyBatchReturnsEmpty() throws {
        let c = try ParityCase.load("tiny-9")
        let f = try FeatureBatch(boardSize: 9, batch: 0, spatial: [], global: [], legal: [])
        let raw = try PackedCPUBackend(model: c.model).evaluateSync(features: f)
        XCTAssertEqual(raw.batch, 0)
        XCTAssertTrue(raw.policyLogits.isEmpty)
        let layers = PackedCPUBackend(model: c.model).layerOutputs(features: f)
        XCTAssertEqual(layers.count, c.model.layers)
        XCTAssertTrue(layers.allSatisfy(\.isEmpty))
    }

    func testUnsupportedBoardSizeRejected() throws {
        let c = try ParityCase.load("tiny-9") // model declares boardSizes [9]
        let f = try FeatureBatch(
            boardSize: 19, batch: 1, spatial: [UInt8](repeating: 0, count: 361 * 32),
            global: [0, 1, 0, 0], legal: [UInt8](repeating: 1, count: 362)
        )
        XCTAssertThrowsError(try PackedCPUBackend(model: c.model).evaluateSync(features: f))
    }

    /// docs/spec/01-network.md §5: "g=15・NOT にも末尾maskを適用する" -- the tail (padding) lanes
    /// of a partial last group must read back as 0 even for gate 15 (true, unconditionally 1) and
    /// gate 3 (NOT a; a reads 0 off-board/from a zero-padded group, so NOT a=1 unconditionally
    /// there too). Uses a hand-built one-layer, one-channel-per-gate model (`@testable import`
    /// gives this test target the internal `LogicModelData` memberwise initializer) so the
    /// padding bits can be inspected directly in the *packed* representation -- unpacking alone
    /// (as the sweep test above does) only ever looks at real lanes and would never catch a
    /// missing tail mask.
    func testTrueAndNotGatePaddingIsMasked() throws {
        let S = 3
        // Hand-built one-layer, two-channel model: channel 0 = gate 15 (true), channel 1 = gate 3
        // (NOT a). Both refs read bank 0 (layer 0's "previous layer" is the 32-channel input),
        // channel 0 vs 1 so A != B. `@testable import` exposes `ModelManifest`/`LogicModelData`'s
        // synthesized (internal) memberwise initializers so this test can build a tiny model
        // in-memory instead of writing a full `.ichigo` directory to disk.
        let manifest = ModelManifest(
            headVersion: 1, boardSizes: [S], channels: 2, layers: 1, dilations: [1], calibrationTemperature: 1.0,
            files: [:], headTensors: [], trainingProvenanceJSON: "{}", rawJSON: "{}"
        )
        let refA = GateReference(bank: 0, channel: 0, dx: 0, dy: 0)
        let refB = GateReference(bank: 0, channel: 1, dx: 0, dy: 0)
        let wiring: [[[GateReference]]] = [[[refA, refB], [refA, refB]]]
        let gates: [[UInt8]] = [[15, 3]]
        let model = LogicModelData(
            manifest: manifest, channels: 2, layers: 1, dilations: [1], wiring: wiring, gates: gates,
            heads: [:], payloadHash: "test"
        )
        let backend = PackedCPUBackend(model: model)
        for B in [1, 2, 31, 33, 63, 65] {
            let f = try syntheticFeatures(boardSize: S, batch: B, seed: UInt64(0xFACE_0000 + B))
            let packed = backend.packedLayerOutputs(features: f)
            let G = PackBits.groupCount(batch: B)
            let lastGroup = G - 1
            let mask = PackBits.validMask(batch: B, group: lastGroup)
            for i in 0 ..< (S * S * 2) {
                let word = packed[0][lastGroup * S * S * 2 + i]
                // The tail (padding) lanes must read 0 regardless of gate, for both channels.
                XCTAssertEqual(word & ~mask, 0, "batch \(B) index \(i): padding lanes not masked for gate")
                // Channel 0 (gate 15, true) is 1 unconditionally, so *every* valid lane must be
                // 1 too -- unlike channel 1 (gate 3, NOT a), whose valid lanes depend on the
                // (real, random) input bit and aren't expected to be all-1.
                if i % 2 == 0 {
                    XCTAssertEqual(word & mask, mask, "batch \(B) index \(i): true-gate valid lanes should all be 1")
                }
            }
        }
    }
}
