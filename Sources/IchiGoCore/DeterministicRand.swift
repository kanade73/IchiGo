/// Only KataGo's fixed-string seeded path is implemented. No OS entropy is used.
struct DeterministicRand {
    private var xorState = [UInt64](repeating: 0, count: 16)
    private var xorIndex = 0
    private var pcgState: UInt64 = 0

    init(seed: String) {
        initialize(seed: seed)
    }

    mutating func initialize(seed: String) {
        let md5 = MD5.hash(Array(seed.utf8))
        let suffix = "|\(md5[0])|\(seed)"
        var counter = 0
        var words: [UInt64] = []
        while words.count < 17 {
            let bytes = SHA256.hash(Array(("\(counter)" + suffix).utf8))
            counter += 37
            for offset in stride(from: 0, to: 32, by: 8) {
                var word: UInt64 = 0
                for byte in bytes[offset ..< offset + 8] {
                    word = (word << 8) | UInt64(byte)
                }
                if word != 0 {
                    words.append(word)
                }
            }
        }
        xorState = Array(words.prefix(16))
        xorIndex = 0
        pcgState = words[16]
    }

    mutating func nextUInt32() -> UInt32 {
        pcgState = pcgState &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let shifted = ((pcgState >> 18) ^ pcgState) >> 27
        let x = UInt32(truncatingIfNeeded: shifted)
        let rotation = Int(pcgState >> 59)
        let pcg = rotation == 0 ? x : (x >> rotation) | (x << (32 - rotation))

        let old = xorState[xorIndex]
        xorIndex = (xorIndex + 1) & 15
        var next = xorState[xorIndex]
        next ^= next << 31
        next ^= next >> 11
        let mixedOld = old ^ (old >> 30)
        xorState[xorIndex] = mixedOld ^ next
        let xor = UInt32(truncatingIfNeeded: (xorState[xorIndex] &* 1_181_783_497_276_652_981) >> 32)
        return pcg &+ xor
    }

    mutating func nextUInt64() -> UInt64 {
        let lower = UInt64(nextUInt32())
        return lower | (UInt64(nextUInt32()) << 32)
    }
}

private enum MD5 {
    private static let shifts: [UInt32] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ]
    private static let constants: [UInt32] = [
        0xD76A_A478, 0xE8C7_B756, 0x2420_70DB, 0xC1BD_CEEE,
        0xF57C_0FAF, 0x4787_C62A, 0xA830_4613, 0xFD46_9501,
        0x6980_98D8, 0x8B44_F7AF, 0xFFFF_5BB1, 0x895C_D7BE,
        0x6B90_1122, 0xFD98_7193, 0xA679_438E, 0x49B4_0821,
        0xF61E_2562, 0xC040_B340, 0x265E_5A51, 0xE9B6_C7AA,
        0xD62F_105D, 0x0244_1453, 0xD8A1_E681, 0xE7D3_FBC8,
        0x21E1_CDE6, 0xC337_07D6, 0xF4D5_0D87, 0x455A_14ED,
        0xA9E3_E905, 0xFCEF_A3F8, 0x676F_02D9, 0x8D2A_4C8A,
        0xFFFA_3942, 0x8771_F681, 0x6D9D_6122, 0xFDE5_380C,
        0xA4BE_EA44, 0x4BDE_CFA9, 0xF6BB_4B60, 0xBEBF_BC70,
        0x289B_7EC6, 0xEAA1_27FA, 0xD4EF_3085, 0x0488_1D05,
        0xD9D4_D039, 0xE6DB_99E5, 0x1FA2_7CF8, 0xC4AC_5665,
        0xF429_2244, 0x432A_FF97, 0xAB94_23A7, 0xFC93_A039,
        0x655B_59C3, 0x8F0C_CC92, 0xFFEF_F47D, 0x8584_5DD1,
        0x6FA8_7E4F, 0xFE2C_E6E0, 0xA301_4314, 0x4E08_11A1,
        0xF753_7E82, 0xBD3A_F235, 0x2AD7_D2BB, 0xEB86_D391,
    ]

    static func hash(_ input: [UInt8]) -> [UInt32] {
        var message = input
        let bitLength = UInt64(message.count) * 8
        message.append(0x80)
        while message.count % 64 != 56 {
            message.append(0)
        }
        for shift in stride(from: 0, through: 56, by: 8) {
            message.append(UInt8(truncatingIfNeeded: bitLength >> shift))
        }

        var state: [UInt32] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476]
        for offset in stride(from: 0, to: message.count, by: 64) {
            var words = [UInt32](repeating: 0, count: 16)
            for index in 0 ..< 16 {
                let start = offset + index * 4
                words[index] = UInt32(message[start]) | (UInt32(message[start + 1]) << 8)
                    | (UInt32(message[start + 2]) << 16) | (UInt32(message[start + 3]) << 24)
            }
            var a = state[0]
            var b = state[1]
            var c = state[2]
            var d = state[3]
            for index in 0 ..< 64 {
                let value: UInt32
                let wordIndex: Int
                switch index {
                case 0 ..< 16:
                    value = (b & c) | (~b & d)
                    wordIndex = index
                case 16 ..< 32:
                    value = (d & b) | (~d & c)
                    wordIndex = (5 * index + 1) % 16
                case 32 ..< 48:
                    value = b ^ c ^ d
                    wordIndex = (3 * index + 5) % 16
                default:
                    value = c ^ (b | ~d)
                    wordIndex = (7 * index) % 16
                }
                let sum = a &+ value &+ constants[index] &+ words[wordIndex]
                let shift = shifts[index]
                let rotated = (sum << shift) | (sum >> (32 - shift))
                (a, b, c, d) = (d, b &+ rotated, b, c)
            }
            state[0] &+= a
            state[1] &+= b
            state[2] &+= c
            state[3] &+= d
        }
        return state
    }
}

private enum SHA256 {
    private static let constants: [UInt32] = [
        0x428A_2F98, 0x7137_4491, 0xB5C0_FBCF, 0xE9B5_DBA5,
        0x3956_C25B, 0x59F1_11F1, 0x923F_82A4, 0xAB1C_5ED5,
        0xD807_AA98, 0x1283_5B01, 0x2431_85BE, 0x550C_7DC3,
        0x72BE_5D74, 0x80DE_B1FE, 0x9BDC_06A7, 0xC19B_F174,
        0xE49B_69C1, 0xEFBE_4786, 0x0FC1_9DC6, 0x240C_A1CC,
        0x2DE9_2C6F, 0x4A74_84AA, 0x5CB0_A9DC, 0x76F9_88DA,
        0x983E_5152, 0xA831_C66D, 0xB003_27C8, 0xBF59_7FC7,
        0xC6E0_0BF3, 0xD5A7_9147, 0x06CA_6351, 0x1429_2967,
        0x27B7_0A85, 0x2E1B_2138, 0x4D2C_6DFC, 0x5338_0D13,
        0x650A_7354, 0x766A_0ABB, 0x81C2_C92E, 0x9272_2C85,
        0xA2BF_E8A1, 0xA81A_664B, 0xC24B_8B70, 0xC76C_51A3,
        0xD192_E819, 0xD699_0624, 0xF40E_3585, 0x106A_A070,
        0x19A4_C116, 0x1E37_6C08, 0x2748_774C, 0x34B0_BCB5,
        0x391C_0CB3, 0x4ED8_AA4A, 0x5B9C_CA4F, 0x682E_6FF3,
        0x748F_82EE, 0x78A5_636F, 0x84C8_7814, 0x8CC7_0208,
        0x90BE_FFFA, 0xA450_6CEB, 0xBEF9_A3F7, 0xC671_78F2,
    ]

    static func hash(_ input: [UInt8]) -> [UInt8] {
        var message = input
        let bitLength = UInt64(message.count) * 8
        message.append(0x80)
        while message.count % 64 != 56 {
            message.append(0)
        }
        for shift in stride(from: 56, through: 0, by: -8) {
            message.append(UInt8(truncatingIfNeeded: bitLength >> shift))
        }

        var state: [UInt32] = [
            0x6A09_E667, 0xBB67_AE85, 0x3C6E_F372, 0xA54F_F53A,
            0x510E_527F, 0x9B05_688C, 0x1F83_D9AB, 0x5BE0_CD19,
        ]
        for offset in stride(from: 0, to: message.count, by: 64) {
            var words = [UInt32](repeating: 0, count: 64)
            for index in 0 ..< 16 {
                let start = offset + index * 4
                words[index] = (UInt32(message[start]) << 24) | (UInt32(message[start + 1]) << 16)
                    | (UInt32(message[start + 2]) << 8) | UInt32(message[start + 3])
            }
            for index in 16 ..< 64 {
                let x = words[index - 15]
                let y = words[index - 2]
                let s0 = rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >> 3)
                let s1 = rotateRight(y, 17) ^ rotateRight(y, 19) ^ (y >> 10)
                words[index] = words[index - 16] &+ s0 &+ words[index - 7] &+ s1
            }
            var a = state[0]
            var b = state[1]
            var c = state[2]
            var d = state[3]
            var e = state[4]
            var f = state[5]
            var g = state[6]
            var h = state[7]
            for index in 0 ..< 64 {
                let sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25)
                let choice = (e & f) ^ (~e & g)
                let temporary1 = h &+ sum1 &+ choice &+ constants[index] &+ words[index]
                let sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22)
                let majority = (a & b) ^ (a & c) ^ (b & c)
                let temporary2 = sum0 &+ majority
                (a, b, c, d, e, f, g, h) = (temporary1 &+ temporary2, a, b, c, d &+ temporary1, e, f, g)
            }
            state[0] &+= a
            state[1] &+= b
            state[2] &+= c
            state[3] &+= d
            state[4] &+= e
            state[5] &+= f
            state[6] &+= g
            state[7] &+= h
        }
        return state.flatMap { word in
            [
                UInt8(word >> 24),
                UInt8(truncatingIfNeeded: word >> 16),
                UInt8(truncatingIfNeeded: word >> 8),
                UInt8(truncatingIfNeeded: word),
            ]
        }
    }

    private static func rotateRight(_ value: UInt32, _ amount: UInt32) -> UInt32 {
        (value >> amount) | (value << (32 - amount))
    }
}
