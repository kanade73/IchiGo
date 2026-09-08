/// KataGo's two-word 128-bit hash. Text form prints `hash1` before `hash0`, matching C++.
public struct Hash128: Equatable, Hashable, Comparable, Sendable, CustomStringConvertible {
    public var hash0: UInt64
    public var hash1: UInt64

    public init(_ hash0: UInt64 = 0, _ hash1: UInt64 = 0) {
        self.hash0 = hash0
        self.hash1 = hash1
    }

    public static func < (lhs: Hash128, rhs: Hash128) -> Bool {
        lhs.hash1 == rhs.hash1 ? lhs.hash0 < rhs.hash0 : lhs.hash1 < rhs.hash1
    }

    public static func ^ (lhs: Hash128, rhs: Hash128) -> Hash128 {
        Hash128(lhs.hash0 ^ rhs.hash0, lhs.hash1 ^ rhs.hash1)
    }

    public static func ^= (lhs: inout Hash128, rhs: Hash128) {
        lhs.hash0 ^= rhs.hash0
        lhs.hash1 ^= rhs.hash1
    }

    public var description: String {
        Self.hex(hash1) + Self.hex(hash0)
    }

    private static func hex(_ value: UInt64) -> String {
        let value = String(value, radix: 16, uppercase: true)
        return String(repeating: "0", count: 16 - value.count) + value
    }
}

enum HashMix {
    static func basicLCong(_ input: UInt64) -> UInt64 {
        2_862_933_555_777_941_757 &* input &+ 3_037_000_493
    }

    static func basicLCong2(_ input: UInt64) -> UInt64 {
        6_364_136_223_846_793_005 &* input &+ 1_442_695_040_888_963_407
    }

    static func murmurMix(_ input: UInt64) -> UInt64 {
        var value = input
        value ^= value >> 33
        value &*= 0xFF51_AFD7_ED55_8CCD
        value ^= value >> 33
        value &*= 0xC4CE_B9FE_1A85_EC53
        value ^= value >> 33
        return value
    }

    static func splitMix64(_ input: UInt64) -> UInt64 {
        var value = input &+ 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    static func rrmxmx(_ input: UInt64) -> UInt64 {
        var value = input ^ input.rotatedRight(49) ^ input.rotatedRight(24)
        value &*= 0x9FB2_1C65_1E98_DF25
        value ^= value >> 28
        value &*= 0x9FB2_1C65_1E98_DF25
        return value ^ (value >> 28)
    }
}

private extension UInt64 {
    func rotatedRight(_ amount: UInt64) -> UInt64 {
        self >> amount | self << (64 - amount)
    }
}
