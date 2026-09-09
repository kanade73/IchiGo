import Foundation
import LogicModel
import XCTest

/// Deterministic, dependency-free PRNG (SplitMix64) for synthesising test bytes.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// docs/spec/05-validation.md §2 bitpack fixtures: `B=0,1,2,31,32,33,63,64,65`, padding bits
/// zero, all-0/all-1 inputs.
final class PackBitsTests: XCTestCase {
    let batches = [0, 1, 2, 31, 32, 33, 63, 64, 65]

    func testGroupCount() {
        let expected: [Int: Int] = [0: 0, 1: 1, 2: 1, 31: 1, 32: 1, 33: 2, 63: 2, 64: 2, 65: 3]
        for (b, g) in expected { XCTAssertEqual(PackBits.groupCount(batch: b), g, "batch \(b)") }
    }

    func testValidMask() {
        // Non-last groups (and any group when B is an exact multiple of 32) are always all-ones;
        // "シフト32は禁止" -- remainder is always in 0...31 here, never shifted by 32.
        XCTAssertEqual(PackBits.validMask(batch: 32, group: 0), 0xFFFF_FFFF)
        XCTAssertEqual(PackBits.validMask(batch: 64, group: 0), 0xFFFF_FFFF)
        XCTAssertEqual(PackBits.validMask(batch: 64, group: 1), 0xFFFF_FFFF)
        XCTAssertEqual(PackBits.validMask(batch: 33, group: 0), 0xFFFF_FFFF) // first group is full
        XCTAssertEqual(PackBits.validMask(batch: 33, group: 1), 0x0000_0001) // remainder 1
        XCTAssertEqual(PackBits.validMask(batch: 31, group: 0), 0x7FFF_FFFF) // remainder 31
        XCTAssertEqual(PackBits.validMask(batch: 63, group: 1), 0x7FFF_FFFF) // remainder 31
        XCTAssertEqual(PackBits.validMask(batch: 65, group: 2), 0x0000_0001) // remainder 1
        XCTAssertEqual(PackBits.validMask(batch: 1, group: 0), 0x0000_0001)
        XCTAssertEqual(PackBits.validMask(batch: 2, group: 0), 0x0000_0003)
    }

    /// Round trip pack -> unpack reproduces the original bytes for every batch size that
    /// straddles a 32-lane boundary, with random 0/1 content.
    func testPackUnpackRoundTrip() {
        let S = 9, C = 32
        for B in batches {
            var rng = SplitMix64(seed: UInt64(0xA11C_E000 + B))
            var bytes = [UInt8](repeating: 0, count: B * S * S * C)
            for i in 0 ..< bytes.count { bytes[i] = UInt8(rng.next() & 1) }
            let packed = PackBits.pack(bytes, boardSize: S, batch: B, channels: C)
            XCTAssertEqual(packed.count, PackBits.groupCount(batch: B) * S * S * C, "batch \(B)")
            let round = PackBits.unpack(packed, boardSize: S, batch: B, channels: C)
            XCTAssertEqual(round, bytes, "batch \(B)")
        }
    }

    func testAllZeroInputPacksToZero() {
        let S = 9, C = 32, B = 65
        let bytes = [UInt8](repeating: 0, count: B * S * S * C)
        let packed = PackBits.pack(bytes, boardSize: S, batch: B, channels: C)
        XCTAssertTrue(packed.allSatisfy { $0 == 0 })
    }

    /// All-ones input: every real lane set, so each group's word equals exactly that group's
    /// valid mask (full groups -> 0xffffffff, the partial last group -> only its low
    /// `remainder` bits).
    func testAllOneInputMatchesValidMaskExactly() {
        let S = 9, C = 4, B = 40 // G=2, remainder=8
        let bytes = [UInt8](repeating: 1, count: B * S * S * C)
        let packed = PackBits.pack(bytes, boardSize: S, batch: B, channels: C)
        let G = PackBits.groupCount(batch: B)
        XCTAssertEqual(G, 2)
        for group in 0 ..< G {
            let expected = PackBits.validMask(batch: B, group: group)
            for i in 0 ..< (S * S * C) {
                XCTAssertEqual(packed[group * S * S * C + i], expected, "group \(group) index \(i)")
            }
        }
    }

    /// Padding lanes (`b >= B`, i.e. bits at position >= remainder in the last group) are never
    /// set by `pack`, regardless of content -- there is nothing there to set. This is the
    /// half of "padding bits zero" that's about storage; `PackedCPUBackendTests` covers the other
    /// half (gate evaluation must also zero them for NOT/true gates, not just leave them
    /// unset-by-construction).
    func testPaddingLanesNeverSetByPack() {
        for B in [1, 2, 31, 33, 63, 65] {
            let S = 9, C = 8
            let bytes = [UInt8](repeating: 1, count: B * S * S * C) // all-ones content
            let packed = PackBits.pack(bytes, boardSize: S, batch: B, channels: C)
            let G = PackBits.groupCount(batch: B)
            let lastGroup = G - 1
            let remainder = B % 32
            guard remainder != 0 else { continue } // exact multiple of 32: no padding lanes
            let mask = PackBits.validMask(batch: B, group: lastGroup)
            for i in 0 ..< (S * S * C) {
                let word = packed[lastGroup * S * S * C + i]
                XCTAssertEqual(word & ~mask, 0, "batch \(B) index \(i): padding lanes set")
            }
        }
    }
}
