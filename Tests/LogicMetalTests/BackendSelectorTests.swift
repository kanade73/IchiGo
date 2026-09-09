import Foundation
import LogicMetal
import XCTest

/// `BackendSelector`/`BackendProfile` (docs/spec/04-tasks.md T25) need no GPU, but live next to
/// `MetalBackend` in the `LogicMetal` target, so their tests live in `LogicMetalTests` too.
final class BackendSelectorTests: XCTestCase {
    func testAutoWithNoProfileFollowsMetalAvailability() {
        XCTAssertEqual(BackendSelector.select(batch: 8, profile: nil, metalAvailable: true), "metal")
        XCTAssertEqual(BackendSelector.select(batch: 8, profile: nil, metalAvailable: false), "cpu")
    }

    func testProfileEntryWins() {
        let profile = BackendProfile(batches: [1: "cpu", 8: "metal", 32: "metal"])
        XCTAssertEqual(BackendSelector.select(batch: 1, profile: profile, metalAvailable: true), "cpu")
        XCTAssertEqual(BackendSelector.select(batch: 8, profile: profile, metalAvailable: true), "metal")
        // No entry for batch 16: falls back to auto, not to a nearby batch's answer.
        XCTAssertEqual(BackendSelector.select(batch: 16, profile: profile, metalAvailable: true), "metal")
        XCTAssertEqual(BackendSelector.select(batch: 16, profile: profile, metalAvailable: false), "cpu")
    }

    func testProfileNamingUnavailableMetalFallsBackToCPU() {
        let profile = BackendProfile(batches: [8: "metal"])
        XCTAssertEqual(BackendSelector.select(batch: 8, profile: profile, metalAvailable: false), "cpu")
    }

    func testFastestPerBatchPicksHigherThroughput() {
        let measurements: [(batch: Int, backend: String, positionsPerSec: Double)] = [
            (1, "cpu", 1200), (1, "metal", 200),
            (8, "cpu", 3000), (8, "metal", 4200),
            (32, "cpu", 3500), (32, "metal", 9000),
        ]
        let winners = BackendSelector.fastestPerBatch(measurements: measurements)
        XCTAssertEqual(winners[1], "cpu")
        XCTAssertEqual(winners[8], "metal")
        XCTAssertEqual(winners[32], "metal")
    }

    func testLoadParsesTheExampleProfile() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("configs/backend-profile.example.json")
        let profile = try BackendProfile.load(path: url.path)
        XCTAssertEqual(profile.hardware, "Apple M5")
        XCTAssertEqual(profile.batches[1], "cpu")
        XCTAssertEqual(profile.batches[8], "metal")
        XCTAssertEqual(profile.batches[32], "metal")
    }

    func testLoadRejectsMissingFile() {
        XCTAssertThrowsError(try BackendProfile.load(path: "/nonexistent/backend-profile.json"))
    }
}
