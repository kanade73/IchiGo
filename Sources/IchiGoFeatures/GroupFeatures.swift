import Foundation
import IchiGoCore

/// featureVersion 2 group-level planes (docs/spec/01-network.md §1, v2 channels 6...15), all from
/// the side to move's view and all exact functions of the current board. They carry chain- and
/// region-level facts that a 2-input logic-gate stack cannot aggregate along a chain by itself
/// (2026-09-23: both CGOS losses were a dead group the model read as alive).
///
/// Bit `i` of each point's `UInt16` is v2 channel `6 + i`:
///
/// | bit | meaning |
/// |---|---|
/// | 0, 1 | own / opponent stones of a chain captured in a ladder (1 liberty: even with the defender moving first; 2 liberties: after some attacker move). KataGo V7 feature 14, split by colour |
/// | 2 | moves that start a working ladder against an opponent 2-liberty chain (KataGo V7 feature 17) |
/// | 3, 4 | own / opponent pass-alive area, stones and territory (Benson; `calculateArea` with no big-territory or non-pass-alive extensions) |
/// | 5, 6 | own / opponent stones of chains with 4 or more liberties (refines the 3+ plane) |
/// | 7, 8 | own / opponent eye regions: empty 4-connected regions of at most `maxEyeRegion` points bordered only by that colour; a single point also has to be a simple (not false) eye |
/// | 9 | stones (either colour) whose chain borders at least two distinct eye regions of its own colour |
public enum GroupFeatures {
    public static let planes = 10
    public static let firstChannel = 6
    public static let maxEyeRegion = 8
    /// featureVersion 3 keeps its `GroupFeaturesV3` planes in bits `v3Shift...` of the same word.
    public static let v3Shift = 10

    /// Every plane depends only on where the stones are (not on ko, history or superko), so the
    /// encoder can rebuild a board from a snapshot's current layout; search and export need no
    /// extra state for featureVersion 2.
    public static func compute(layout: StoneLayout, size: Int, toMove: Player) -> [UInt16] {
        compute(board: board(layout: layout, size: size), toMove: toMove)
    }

    /// Group planes for a whole batch, positions in parallel (each builds its own `Board`); the
    /// encoder ran them one by one on the evaluator's thread.
    public static func compute(batch snapshots: [PositionSnapshot], featureVersion: Int = 2, minParallel: Int = 8) -> [[UInt16]] {
        if snapshots.count < minParallel {
            return snapshots.map { compute(snapshot: $0, featureVersion: featureVersion) }
        }
        let out = ResultSlots(count: snapshots.count)
        DispatchQueue.concurrentPerform(iterations: snapshots.count) { i in
            out.set(i, compute(snapshot: snapshots[i], featureVersion: featureVersion))
        }
        return out.values
    }

    /// The v2 planes (bits 0...9), plus the v3 planes in bits `v3Shift...` for featureVersion >= 3.
    public static func compute(snapshot snap: PositionSnapshot, featureVersion: Int) -> [UInt16] {
        var bits = compute(layout: snap.layouts[0], size: snap.boardSize, toMove: snap.toMove)
        if featureVersion >= 3 {
            let v3 = GroupFeaturesV3.compute(layout: snap.layouts[0], size: snap.boardSize)
            for p in bits.indices where v3[p] != 0 { bits[p] |= UInt16(v3[p]) << UInt16(v3Shift) }
        }
        return bits
    }

    /// Each index is written by exactly one `concurrentPerform` iteration and read only after it
    /// returns.
    private final class ResultSlots: @unchecked Sendable {
        private let slots: UnsafeMutableBufferPointer<[UInt16]>
        init(count: Int) {
            slots = .allocate(capacity: count)
            slots.initialize(repeating: [])
        }
        deinit {
            slots.deinitialize()
            slots.deallocate()
        }
        func set(_ i: Int, _ v: [UInt16]) { slots[i] = v }
        var values: [[UInt16]] { Array(slots) }
    }

    /// One bulk placement and one chain rebuild (placing stones one at a time rebuilt every chain
    /// per stone: 46% of the encoder's time in search). For a legal layout, every chain keeps a
    /// liberty, so nothing is captured and the result equals sequential `setStone`.
    static func board(layout: StoneLayout, size: Int) -> Board {
        let board = Board(size, size)
        var placements: [(loc: Loc, color: Color)] = []
        placements.reserveCapacity(size * size)
        for p in 0 ..< size * size where layout[p] != 0 {
            placements.append((loc: Location.getLoc(p % size, p / size, size), color: Color(rawValue: Int8(layout[p])) ?? .empty))
        }
        board.setStonesTolerant(placements)
        return board
    }

    public static func compute(board: Board, toMove: Player) -> [UInt16] {
        let S = board.xSize
        var bits = [UInt16](repeating: 0, count: S * S)
        let me = toMove.color
        let opp = toMove.opponent.color
        @inline(__always) func point(_ loc: Loc) -> Int { Location.getY(loc, S) * S + Location.getX(loc, S) }
        @inline(__always) func mark(_ loc: Loc, _ bit: Int) { bits[point(loc)] |= UInt16(1) << UInt16(bit) }
        let locs = board.playableLocations()

        // ladders (bits 0-2)
        let scratch = board.copy()
        var solved: [Loc: (captured: Bool, working: [Loc])] = [:]
        var buffer: [Loc] = []
        for loc in locs {
            let c = board.colors[loc]
            guard c == .black || c == .white else { continue }
            let libs = board.getNumLiberties(loc)
            guard libs == 1 || libs == 2 else { continue }
            let head = board.chainHead[loc]
            let result: (captured: Bool, working: [Loc])
            if let r = solved[head] {
                result = r
            } else {
                var working: [Loc] = []
                let captured = libs == 1
                    ? scratch.searchIsLadderCaptured(loc, defenderFirst: true, buffer: &buffer)
                    : scratch.searchIsLadderCapturedAttackerFirst2Libs(loc, buffer: &buffer, workingMoves: &working)
                result = (captured, captured ? working : [])
                solved[head] = result
            }
            guard result.captured else { continue }
            mark(loc, c == me ? 0 : 1)
            if c == opp { for w in result.working { mark(w, 2) } }
        }

        // pass-alive area (bits 3-4)
        let area = board.calculateArea(nonPassAliveStones: false, safeBigTerritories: false, unsafeBigTerritories: false)
        for loc in locs {
            if area[loc] == me { mark(loc, 3) } else if area[loc] == opp { mark(loc, 4) }
        }

        // 4+ liberty chains (bits 5-6)
        for loc in locs {
            let c = board.colors[loc]
            guard c == .black || c == .white, board.getNumLiberties(loc) >= 4 else { continue }
            mark(loc, c == me ? 5 : 6)
        }

        // eye regions (bits 7-8) and two-eye chains (bit 9)
        var regionOf = [Int](repeating: -1, count: Board.maxArrSize)
        var eyeRegionsByChain: [Loc: Set<Int>] = [:]
        var regionCount = 0
        for start in locs where board.colors[start] == .empty && regionOf[start] < 0 {
            let id = regionCount
            regionCount += 1
            var stack = [start]
            var members: [Loc] = []
            var borderColors = Set<Int8>()
            var borderHeads = Set<Loc>()
            regionOf[start] = id
            while let loc = stack.popLast() {
                members.append(loc)
                for offset in board.adjOffsets.prefix(4) {
                    let n = loc + offset
                    switch board.colors[n] {
                    case .empty where regionOf[n] < 0:
                        regionOf[n] = id
                        stack.append(n)
                    case .black, .white:
                        borderColors.insert(board.colors[n].rawValue)
                        borderHeads.insert(board.chainHead[n])
                    default:
                        break
                    }
                }
            }
            guard members.count <= maxEyeRegion, borderColors.count == 1,
                  let owner = Color(rawValue: borderColors.first!)?.player else { continue }
            if members.count == 1, !board.isSimpleEye(members[0], owner) { continue }
            for m in members { mark(m, owner.color == me ? 7 : 8) }
            for h in borderHeads { eyeRegionsByChain[h, default: []].insert(id) }
        }
        for loc in locs {
            let c = board.colors[loc]
            guard c == .black || c == .white, (eyeRegionsByChain[board.chainHead[loc]]?.count ?? 0) >= 2 else { continue }
            mark(loc, 9)
        }
        return bits
    }
}
