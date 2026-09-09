import Foundation
import LogicMetal
import LogicModel
import XCTest

/// `make check-metal`: MetalBackend lifecycle and batch-size sweep (docs/spec/05-validation.md
/// §2 bitpack row: "B=0,1,2,31,32,33,63,64,65" -- this ticket (T22) is byte-per-activation, not
/// bit-packed, but the batch sizes the task calls for are the ones that straddle a 32-lane pack
/// boundary anyway: 0,1,2,31,32,33,64).
final class MetalBackendTests: XCTestCase {
    func testBatchSizeSweepMatchesScalarBackend() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9")
        let metal = try MetalBackend(model: c.model)
        let scalar = ScalarBackend(model: c.model)

        for b in [0, 1, 2, 31, 32, 33, 64] {
            let features = try syntheticFeatures(boardSize: c.boardSize, batch: b, seed: UInt64(0x5EED_0000 + b))

            let metalLayers = try await metal.layerOutputs(features: features)
            let scalarLayers = scalar.layerOutputs(features: features)
            XCTAssertEqual(metalLayers.count, scalarLayers.count, "batch \(b): layer count")
            for l in 0 ..< scalarLayers.count {
                XCTAssertEqual(metalLayers[l], scalarLayers[l], "batch \(b): layer \(l) bits differ")
            }

            let rawMetal = try await metal.evaluate(features: features)
            let rawScalar = try scalar.evaluateSync(features: features)
            XCTAssertEqual(rawMetal.batch, b)
            assertHeadClose(rawMetal.policyLogits, rawScalar.policyLogits, "batch \(b) policyLogits")
            assertHeadClose(rawMetal.wdlLogits, rawScalar.wdlLogits, "batch \(b) wdlLogits")
            assertHeadClose(rawMetal.scoreMean, rawScalar.scoreMean, "batch \(b) scoreMean")
            assertHeadClose(rawMetal.ownership, rawScalar.ownership, "batch \(b) ownership")
        }
    }

    func testEmptyBatchReturnsEmpty() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9")
        let metal = try MetalBackend(model: c.model)
        let f = try FeatureBatch(boardSize: 9, batch: 0, spatial: [], global: [], legal: [])
        let raw = try await metal.evaluate(features: f)
        XCTAssertEqual(raw.batch, 0)
        XCTAssertTrue(raw.policyLogits.isEmpty)
        let layers = try await metal.layerOutputs(features: f)
        XCTAssertEqual(layers.count, c.model.layers)
        XCTAssertTrue(layers.allSatisfy(\.isEmpty))
    }

    /// "検収: ...GPU欠如/command buffer errorの失敗動作" -- a model whose board size the backend
    /// was not built for must throw, not crash or silently return garbage (mirrors
    /// `ParityTests.testUnsupportedBoardSizeRejected` for the CPU backend).
    func testUnsupportedBoardSizeThrows() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9") // model declares boardSizes [9]
        let metal = try MetalBackend(model: c.model)
        let f = try FeatureBatch(
            boardSize: 19, batch: 1,
            spatial: [UInt8](repeating: 0, count: 361 * 32), global: [0, 1, 0, 0],
            legal: [UInt8](repeating: 1, count: 362)
        )
        do {
            _ = try await metal.evaluate(features: f)
            XCTFail("expected an error for an unsupported board size")
        } catch let e as LogicModelError {
            guard case .invalidInput = e else { return XCTFail("expected .invalidInput, got \(e)") }
        }
    }

    func testInitReportsBackendUnavailableCleanlyWhenNoDevice() throws {
        // We cannot force `MTLCreateSystemDefaultDevice()` to return nil on a host that has a
        // device, so this only documents/exercises the error path's shape when it *can* run: on
        // a host without Metal, `MetalBackend.init` itself must throw `.backendUnavailable`
        // rather than crash. Skip (rather than assert nothing) on hosts with a device, since
        // `requireMetal()`-style skip would defeat the point here.
        guard !MetalAvailability.probe().available else {
            throw XCTSkip("this host has a Metal device; the no-device failure path is exercised on CI hosts without one")
        }
        let c = try ParityCase.load("tiny-9")
        XCTAssertThrowsError(try MetalBackend(model: c.model)) { error in
            guard let e = error as? LogicModelError, case .backendUnavailable = e else {
                return XCTFail("expected .backendUnavailable, got \(error)")
            }
        }
    }

    func testDeterministicAcrossRepeatedRuns() async throws {
        try requireMetal()
        let c = try ParityCase.load("tiny-9")
        let metal = try MetalBackend(model: c.model)
        let first = try await metal.layerOutputs(features: c.features)
        let second = try await metal.layerOutputs(features: c.features)
        XCTAssertEqual(first, second)
    }

    func testLUT4IsRejectedByBothMetalBackends() throws {
        let c = try ParityCase.load("tiny-9-lut4")
        for make in [
            { try MetalBackend(model: c.model) as any LogicBackend },
            { try MetalPackedBackend(model: c.model) as any LogicBackend },
        ] {
            XCTAssertThrowsError(try make()) { error in
                guard let e = error as? LogicModelError, case let .backendUnavailable(message) = e else {
                    return XCTFail("expected .backendUnavailable, got \(error)")
                }
                XCTAssertTrue(message.contains("gate arity 4"))
            }
        }
    }
}
