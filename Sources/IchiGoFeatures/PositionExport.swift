import Foundation
import IchiGoCore

/// `ichigo features` support (docs/spec/02-training.md §2): replay an SGF main line under the v1
/// rules, and emit one JSON row per intermediate position with the encoded features.
///
/// `positionId` = SHA256 of canonical JSON (sorted keys, no whitespace, UTF-8) of
/// `{boardSize, komi, rulesId, initialStones, initialPlayer, moves}`; `gameId` = same over the
/// whole game (all moves). `initialStones` are sorted by colour, then y, then x.
public enum PositionExport {
    public struct GameReplay {
        public let boardSize: Int
        public let komi: Float
        public let initialStones: [[String]]   // [["B","aa"], ...] sorted
        public let initialPlayer: String       // "B"/"W"
        public let moves: [[String]]           // [["B","E5"], ["W","pass"], ...] GTP
        public let snapshots: [PositionSnapshot]   // one per turn 0...moves.count
        public let gameId: String
        public let result: String?
    }

    public enum RejectReason: Error, CustomStringConvertible {
        case parse(String)
        case boardSize(Int, expected: Int)
        case illegalMove(index: Int, player: String, move: String, reason: String)
        case invalidInitialStone(String)
        case komi(Float)

        public var description: String {
            switch self {
            case let .parse(m): "parse: \(m)"
            case let .boardSize(s, e): "board size \(s) != \(e)"
            case let .illegalMove(i, p, m, r): "illegal move \(i) (\(p) \(m)): \(r)"
            case let .invalidInitialStone(s): "invalid initial stone \(s)"
            case let .komi(k): "invalid komi \(k)"
            }
        }
    }

    /// Replays under `cgos-area-psk-v1` (the SGF `RU` tag is informational only). Rejects
    /// size mismatch, unsupported setup, suicide and superko violations with the move number.
    public static func replay(sgf: String, expectedSize: Int, komiOverride: Float? = nil) throws -> GameReplay {
        let game: SGFGame
        do { game = try SGFReader.parse(sgf) } catch { throw RejectReason.parse("\(error)") }
        guard game.boardXSize == expectedSize, game.boardYSize == expectedSize else {
            throw RejectReason.boardSize(game.boardXSize, expected: expectedSize)
        }
        let S = expectedSize
        let komi = komiOverride ?? game.komi
        var stones: [(player: Player, x: Int, y: Int)] = []
        for p in game.placements {
            stones.append((p.pla, Location.getX(p.loc, S), Location.getY(p.loc, S)))
        }
        stones.sort { a, b in
            if a.player != b.player { return a.player == .black }
            if a.y != b.y { return a.y < b.y }
            return a.x < b.x
        }
        let onlyBlack = !stones.isEmpty && stones.allSatisfy { $0.player == .black }
        var initialPlayer: Player = game.playerToMove ?? (onlyBlack ? .white : .black)
        if let first = game.moves.first { initialPlayer = first.pla }
        let state: GameState
        do {
            state = try GameState(boardSize: S, komi: komi, initialStones: stones, initialPlayer: initialPlayer)
        } catch GameStateError.invalidKomi(let k) {
            throw RejectReason.komi(k)
        } catch {
            throw RejectReason.invalidInitialStone("\(error)")
        }
        var snapshots = [state.snapshot()]
        var moves: [[String]] = []
        for (i, m) in game.moves.enumerated() {
            let coord = Coordinates.move(fromLoc: m.loc, size: S)
            let colour = m.pla == .black ? "B" : "W"
            let gtp = Coordinates.gtpString(coord, size: S)
            do { try state.play(m.pla, coord) } catch {
                throw RejectReason.illegalMove(index: i, player: colour, move: gtp, reason: "\(error)")
            }
            moves.append([colour, gtp])
            snapshots.append(state.snapshot())
        }
        let initialStrings = stones.map { [$0.player == .black ? "B" : "W", Coordinates.sgfString(.point(x: $0.x, y: $0.y), size: S)] }
        let ip = initialPlayer == .black ? "B" : "W"
        let gameId = canonicalHash(boardSize: S, komi: komi, initialStones: initialStrings, initialPlayer: ip, moves: moves)
        return GameReplay(boardSize: S, komi: komi, initialStones: initialStrings, initialPlayer: ip, moves: moves, snapshots: snapshots, gameId: gameId, result: game.result)
    }

    /// SHA256 hex of the canonical JSON `{boardSize,initialPlayer,initialStones,komi,moves,rulesId}`.
    public static func canonicalHash(boardSize: Int, komi: Float, initialStones: [[String]], initialPlayer: String, moves: [[String]]) -> String {
        let obj: [String: Any] = [
            "boardSize": boardSize, "komi": komiNumber(komi), "rulesId": IchiGoRules.rulesID,
            "initialStones": initialStones, "initialPlayer": initialPlayer, "moves": moves,
        ]
        let data = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])
        return SHA256Hex.digest(data)
    }

    /// Komi serialised as an integer when integral (e.g. 7) or a half (7.5), matching Python `json.dumps`.
    static func komiNumber(_ komi: Float) -> Any {
        komi == komi.rounded() ? Int(komi) : Double(komi)
    }

    /// One JSONL row (docs/spec/02-training.md §2) for turn `turn` of `replay`.
    public static func row(_ replay: GameReplay, turn: Int) throws -> [String: Any] {
        let snap = replay.snapshots[turn]
        let enc = try FeatureEncoder.encode([snap])
        let moves = Array(replay.moves.prefix(turn))
        let pid = canonicalHash(boardSize: replay.boardSize, komi: replay.komi, initialStones: replay.initialStones, initialPlayer: replay.initialPlayer, moves: moves)
        return [
            "schemaVersion": 1,
            "positionId": pid,
            "gameId": replay.gameId,
            "boardSize": replay.boardSize,
            "komi": komiNumber(replay.komi),
            "rulesId": IchiGoRules.rulesID,
            "initialStones": replay.initialStones,
            "initialPlayer": replay.initialPlayer,
            "moves": moves,
            "turnNumber": turn,
            "toMove": snap.toMove == .black ? "B" : "W",
            "spatial": enc.spatial.map { Int($0) },
            "global": enc.global.map { Double($0) },
            "legal": enc.legal.map { Int($0) },
        ]
    }
}

/// Minimal SHA-256 (duplicate of LogicModel.SHA256 to keep IchiGoFeatures free of LogicModel).
enum SHA256Hex {
    private static let k: [UInt32] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
        0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
        0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
        0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
        0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ]

    static func digest(_ data: Data) -> String {
        var h: [UInt32] = [0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19]
        var msg = [UInt8](data)
        let bitLen = UInt64(msg.count) * 8
        msg.append(0x80)
        while msg.count % 64 != 56 { msg.append(0) }
        for i in (0 ..< 8).reversed() { msg.append(UInt8((bitLen >> (UInt64(i) * 8)) & 0xff)) }
        var w = [UInt32](repeating: 0, count: 64)
        var chunk = 0
        while chunk < msg.count {
            for i in 0 ..< 16 {
                let j = chunk + i * 4
                w[i] = UInt32(msg[j]) << 24 | UInt32(msg[j + 1]) << 16 | UInt32(msg[j + 2]) << 8 | UInt32(msg[j + 3])
            }
            for i in 16 ..< 64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
                w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
            }
            var a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7]
            for i in 0 ..< 64 {
                let t1 = hh &+ (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) &+ ((e & f) ^ (~e & g)) &+ k[i] &+ w[i]
                let t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) &+ ((a & b) ^ (a & c) ^ (b & c))
                hh = g; g = f; f = e; e = d &+ t1; d = c; c = b; b = a; a = t1 &+ t2
            }
            h[0] = h[0] &+ a; h[1] = h[1] &+ b; h[2] = h[2] &+ c; h[3] = h[3] &+ d
            h[4] = h[4] &+ e; h[5] = h[5] &+ f; h[6] = h[6] &+ g; h[7] = h[7] &+ hh
            chunk += 64
        }
        return h.map { String(format: "%08x", $0) }.joined()
    }

    @inline(__always) private static func rotr(_ x: UInt32, _ n: UInt32) -> UInt32 { (x >> n) | (x << (32 - n)) }
}
