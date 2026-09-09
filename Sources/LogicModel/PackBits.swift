import Foundation

/// Batch-direction bit packing (docs/spec/01-network.md §5, docs/spec/04-tasks.md T23).
///
/// Packs a `[B,S,S,C]` byte tensor (values exactly 0/1) into `[G,S,S,C]` `UInt32` words, where
/// `G = ceil(B/32)` and bit `k` of `packed[((group*S+y)*S+x)*C+c]` is the batch sample
/// `b = group*32+k`. Batch is packed -- never spatial position or channel. This is purely a
/// storage transform; every backend (`PackedCPUBackend`, `MetalPackedBackend`) that consumes
/// packed tensors must reproduce `ScalarBackend` bit-for-bit once unpacked.
public enum PackBits {
    /// `G = ceil(B/32)`. `B=0` packs to zero groups.
    public static func groupCount(batch B: Int) -> Int { (B + 31) / 32 }

    /// Packs `[B,S,S,C]` bytes (0/1) into `[G,S,S,C]` `UInt32` words. Padding lanes
    /// (`b >= B`, i.e. the high bits of the last group when `B` is not a multiple of 32) are left
    /// zero -- there is nothing to pack into them, and `validMask` is what marks them invalid for
    /// gate evaluation.
    public static func pack(_ bytes: [UInt8], boardSize S: Int, batch B: Int, channels C: Int) -> [UInt32] {
        let G = groupCount(batch: B)
        var out = [UInt32](repeating: 0, count: G * S * S * C)
        guard B > 0 else { return out }
        precondition(bytes.count == B * S * S * C, "PackBits.pack: byte count \(bytes.count) != B*S*S*C")
        bytes.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { packed in
                for b in 0 ..< B {
                    let group = b / 32
                    let bit: UInt32 = 1 << UInt32(b % 32)
                    let outGroupBase = group * S * S * C
                    let inBatchBase = b * S * S * C
                    for i in 0 ..< (S * S * C) where input[inBatchBase + i] != 0 {
                        packed[outGroupBase + i] |= bit
                    }
                }
            }
        }
        return out
    }

    /// Inverse of `pack`: unpacks `[G,S,S,C]` `UInt32` words back to `[B,S,S,C]` bytes (0/1),
    /// dropping the padding lanes beyond `batch`.
    public static func unpack(_ packed: [UInt32], boardSize S: Int, batch B: Int, channels C: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: B * S * S * C)
        guard B > 0 else { return out }
        let G = groupCount(batch: B)
        precondition(packed.count == G * S * S * C, "PackBits.unpack: word count \(packed.count) != G*S*S*C")
        packed.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { o in
                for b in 0 ..< B {
                    let group = b / 32
                    let k = UInt32(b % 32)
                    let inGroupBase = group * S * S * C
                    let outBatchBase = b * S * S * C
                    for i in 0 ..< (S * S * C) {
                        o[outBatchBase + i] = UInt8((p[inGroupBase + i] >> k) & 1)
                    }
                }
            }
        }
        return out
    }

    /// The valid-lane mask for one group's word (docs/spec/01-network.md §5):
    /// `valid = remainder==0 ? 0xffffffff : ((1u<<remainder)-1u)`, applied only to the *last*
    /// group -- every earlier group is fully populated (32/32 real samples) and its mask is all
    /// ones. `remainder` is always in `0...31` here, so the shift is never by 32
    /// ("シフト32は禁止").
    public static func validMask(batch B: Int, group: Int) -> UInt32 {
        let G = groupCount(batch: B)
        guard group == G - 1 else { return 0xFFFF_FFFF }
        let remainder = B % 32
        return remainder == 0 ? 0xFFFF_FFFF : (UInt32(1) << UInt32(remainder)) - 1
    }
}
