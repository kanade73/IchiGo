import Foundation

/// featureVersion 3 planes (docs/spec/01-network.md §1): four chain/group facts, set on stones of
/// either colour (channels 0/1 give the colour), independent of the side to move. Chosen with
/// probes over featureVersion 2 models and confirmed on unused games (docs/implementation-status.md
/// 2026-09-25). A *group* is a set of same-colour chains joined when two chains share at least two
/// liberties (a miai connection).
///
/// | bit | channel | meaning |
/// |---|---|---|
/// | 0 | 4 | the stone's group has at least `groupLibertyThreshold` (8) liberties |
/// | 1 | 5 | the stone's group has an eye credit of at least 2 (see below) |
/// | 2 | 16 | the stone's chain touches an enemy chain with more liberties than itself |
/// | 3 | 27 | the stone's chain (at most 3 liberties) can be captured with the attacker to move: a capture search of depth `readerDepth` and at most `readerBudget` nodes finds a capture (ko is ignored; an unfinished search counts as no capture) |
///
/// Eye credit: every empty region bordered by one colour only counts for that colour: a single
/// point that is a false eye 0; more than `maxEyeRegion` points 2; otherwise by shape: at most 2
/// points 1, 7 or more points 2, else 2 if at least two of its points each split it when occupied,
/// 1 if exactly one does (the vital point, counted as the opponent's to take), and with no such
/// point 2 for 6 points and 1 otherwise. A group sums the credit of the distinct regions that
/// contain one of its liberties.
public enum GroupFeaturesV3 {
    public static let planes = 4
    /// v2 channels these replace: history t=2 (4, 5), empty (16), edge ring (27).
    public static let channels = [4, 5, 16, 27]
    public static let groupLibertyThreshold = 8
    public static let readerDepth = 10
    public static let readerBudget = 600
    public static let maxEyeRegion = 8

    struct Geometry: Sendable {
        let size: Int
        let adjacent: [Int]      // 4 per point, -1 off board
        let diagonal: [Int]      // 4 per point, -1 off board
        let offBoardDiagonals: [Int]

        init(_ S: Int) {
            size = S
            var adj = [Int](repeating: -1, count: S * S * 4)
            var diag = [Int](repeating: -1, count: S * S * 4)
            var off = [Int](repeating: 0, count: S * S)
            for p in 0 ..< S * S {
                let x = p % S, y = p / S
                let a = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                let d = [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]
                for k in 0 ..< 4 {
                    if a[k].0 >= 0, a[k].0 < S, a[k].1 >= 0, a[k].1 < S { adj[p * 4 + k] = a[k].1 * S + a[k].0 }
                    if d[k].0 >= 0, d[k].0 < S, d[k].1 >= 0, d[k].1 < S { diag[p * 4 + k] = d[k].1 * S + d[k].0 } else { off[p] += 1 }
                }
            }
            adjacent = adj
            diagonal = diag
            offBoardDiagonals = off
        }
    }

    static let geometry9 = Geometry(9)
    static let geometry19 = Geometry(19)

    static func geometry(_ S: Int) -> Geometry {
        S == 9 ? geometry9 : S == 19 ? geometry19 : Geometry(S)
    }

    /// `layout`: 0 empty, 1 black, 2 white (row-major). Returns bit `i` = plane `i` per point.
    public static func compute(layout: StoneLayout, size S: Int) -> [UInt8] {
        let g = geometry(S)
        let P = S * S
        var out = [UInt8](repeating: 0, count: P)
        var analysis = Analysis(layout: layout, geometry: g)
        analysis.findChains()
        analysis.findRegions()
        let credit = analysis.regionCredit()

        // groups: union-find over same-colour chains sharing >= 2 liberties
        let n = analysis.chainColor.count
        var parent = Array(0 ..< n)
        func find(_ i: Int) -> Int {
            var i = i
            while parent[i] != i {
                parent[i] = parent[parent[i]]
                i = parent[i]
            }
            return i
        }
        let W = analysis.words
        for i in 0 ..< n {
            for j in (i + 1) ..< n where analysis.chainColor[i] == analysis.chainColor[j] {
                var shared = 0
                for w in 0 ..< W { shared += (analysis.chainLibBits[i * W + w] & analysis.chainLibBits[j * W + w]).nonzeroBitCount }
                if shared >= 2 { parent[find(i)] = find(j) }
            }
        }
        var groupBits = [UInt64](repeating: 0, count: n * W)
        for i in 0 ..< n {
            let r = find(i)
            for w in 0 ..< W { groupBits[r * W + w] |= analysis.chainLibBits[i * W + w] }
        }
        var groupLibs = [Int](repeating: 0, count: n)
        var groupCredit = [Int](repeating: 0, count: n)
        var seenRegion = [Int](repeating: -1, count: analysis.regionSize.count)
        for r in 0 ..< n where find(r) == r {
            var libs = 0, eyes = 0
            for w in 0 ..< W {
                var bits = groupBits[r * W + w]
                libs += bits.nonzeroBitCount
                while bits != 0 {
                    let q = w * 64 + bits.trailingZeroBitCount
                    bits &= bits - 1
                    let reg = analysis.regionOf[q]
                    if seenRegion[reg] != r {
                        seenRegion[reg] = r
                        if analysis.regionOwner[reg] == analysis.chainColor[r] { eyes += credit[reg] }
                    }
                }
            }
            groupLibs[r] = libs
            groupCredit[r] = eyes
        }

        var reader = Reader(geometry: g, depth: readerDepth, budget: readerBudget)
        for c in 0 ..< n {
            let r = find(c)
            var bits: UInt8 = 0
            if groupLibs[r] >= groupLibertyThreshold { bits |= 1 }
            if groupCredit[r] >= 2 { bits |= 2 }
            if analysis.chainTouchesStrongerEnemy(c) { bits |= 4 }
            if analysis.chainLibs[c] <= 3, reader.canCapture(layout, target: analysis.chainFirstStone[c]) { bits |= 8 }
            guard bits != 0 else { continue }
            for p in analysis.chainStones(c) { out[p] = bits }
        }
        return out
    }

    // MARK: - chains and regions

    struct Analysis {
        let layout: StoneLayout
        let g: Geometry
        let P: Int
        let words: Int
        var chainOf: [Int]
        var chainColor: [UInt8] = []
        var chainLibs: [Int] = []
        var chainFirstStone: [Int] = []
        var chainLibBits: [UInt64] = []
        var chainNext: [Int]             // stones of a chain as a linked list from chainFirstStone
        var regionOf: [Int]
        var regionSize: [Int] = []
        var regionOwner: [UInt8] = []    // 0 unless bordered by exactly one colour
        var regionFirst: [Int] = []
        var regionNext: [Int]

        init(layout: StoneLayout, geometry: Geometry) {
            self.layout = layout
            g = geometry
            P = geometry.size * geometry.size
            words = (P + 63) / 64
            chainOf = [Int](repeating: -1, count: P)
            chainNext = [Int](repeating: -1, count: P)
            regionOf = [Int](repeating: -1, count: P)
            regionNext = [Int](repeating: -1, count: P)
        }

        mutating func findChains() {
            var stack: [Int] = []
            for start in 0 ..< P where layout[start] != 0 && chainOf[start] < 0 {
                let id = chainColor.count
                let color = layout[start]
                var libBits = [UInt64](repeating: 0, count: words)
                var last = -1
                chainOf[start] = id
                stack.append(start)
                chainFirstStone.append(start)
                while let p = stack.popLast() {
                    if last >= 0 { chainNext[last] = p }
                    last = p
                    for k in 0 ..< 4 {
                        let q = g.adjacent[p * 4 + k]
                        guard q >= 0 else { continue }
                        if layout[q] == 0 {
                            libBits[q >> 6] |= UInt64(1) << UInt64(q & 63)
                        } else if layout[q] == color, chainOf[q] < 0 {
                            chainOf[q] = id
                            stack.append(q)
                        }
                    }
                }
                chainColor.append(color)
                chainLibs.append(libBits.reduce(0) { $0 + $1.nonzeroBitCount })
                chainLibBits.append(contentsOf: libBits)
            }
        }

        func chainStones(_ c: Int) -> [Int] {
            var out: [Int] = []
            var p = chainFirstStone[c]
            while p >= 0 {
                out.append(p)
                p = chainNext[p]
            }
            return out
        }

        func chainTouchesStrongerEnemy(_ c: Int) -> Bool {
            let own = chainLibs[c]
            for p in chainStones(c) {
                for k in 0 ..< 4 {
                    let q = g.adjacent[p * 4 + k]
                    guard q >= 0, layout[q] != 0, layout[q] != chainColor[c] else { continue }
                    if chainLibs[chainOf[q]] > own { return true }
                }
            }
            return false
        }

        mutating func findRegions() {
            var stack: [Int] = []
            for start in 0 ..< P where layout[start] == 0 && regionOf[start] < 0 {
                let id = regionSize.count
                var size = 0
                var border: UInt8 = 0
                var last = -1
                regionOf[start] = id
                stack.append(start)
                regionFirst.append(start)
                while let p = stack.popLast() {
                    size += 1
                    if last >= 0 { regionNext[last] = p }
                    last = p
                    for k in 0 ..< 4 {
                        let q = g.adjacent[p * 4 + k]
                        guard q >= 0 else { continue }
                        if layout[q] == 0 {
                            if regionOf[q] < 0 {
                                regionOf[q] = id
                                stack.append(q)
                            }
                        } else {
                            border |= layout[q] == 1 ? 1 : 2
                        }
                    }
                }
                regionSize.append(size)
                regionOwner.append(border == 1 ? 1 : border == 2 ? 2 : 0)
            }
        }

        func regionPoints(_ r: Int) -> [Int] {
            var out: [Int] = []
            var p = regionFirst[r]
            while p >= 0 {
                out.append(p)
                p = regionNext[p]
            }
            return out
        }

        /// Eye credit of every region for its owner (0 for regions with no single owner).
        func regionCredit() -> [Int] {
            var credit = [Int](repeating: 0, count: regionSize.count)
            for r in 0 ..< regionSize.count where regionOwner[r] != 0 {
                let size = regionSize[r]
                if size == 1 {
                    credit[r] = realSingleEye(regionFirst[r], owner: regionOwner[r]) ? 1 : 0
                } else if size >= 7 {   // includes regions over maxEyeRegion
                    credit[r] = 2
                } else if size <= 2 {
                    credit[r] = 1
                } else {
                    let cuts = splittingPoints(regionPoints(r))
                    credit[r] = cuts >= 2 ? 2 : cuts == 1 ? 1 : (size == 6 ? 2 : 1)
                }
            }
            return credit
        }

        func realSingleEye(_ p: Int, owner: UInt8) -> Bool {
            let enemy: UInt8 = owner == 1 ? 2 : 1
            var bad = 0
            for k in 0 ..< 4 {
                let q = g.diagonal[p * 4 + k]
                if q >= 0, layout[q] == enemy { bad += 1 }
            }
            return g.offBoardDiagonals[p] > 0 ? bad == 0 : bad <= 1
        }

        /// Points of a region whose removal leaves the rest disconnected.
        func splittingPoints(_ pts: [Int]) -> Int {
            var cuts = 0
            for removed in pts {
                let rest = pts.filter { $0 != removed }
                var seen: Set<Int> = [rest[0]]
                var stack = [rest[0]]
                while let p = stack.popLast() {
                    for k in 0 ..< 4 {
                        let q = g.adjacent[p * 4 + k]
                        if q >= 0, q != removed, regionOf[q] == regionOf[p], layout[q] == 0, seen.insert(q).inserted { stack.append(q) }
                    }
                }
                if seen.count < rest.count { cuts += 1 }
            }
            return cuts
        }
    }

    // MARK: - capture search

    /// Depth-limited capture search on a plain colour array (ko ignored). Attacker moves are the
    /// target's liberties, most-open first; defender moves are its liberties and the liberties of
    /// adjacent enemy chains with at most 2 liberties. Mirrors the probe prototype that selected
    /// this feature.
    struct Reader {
        let g: Geometry
        let depth: Int
        let budget: Int
        private var remaining = 0
        private var mark: [UInt32]
        private var stamp: UInt32 = 0

        init(geometry: Geometry, depth: Int, budget: Int) {
            g = geometry
            self.depth = depth
            self.budget = budget
            mark = [UInt32](repeating: 0, count: geometry.size * geometry.size)
        }

        mutating func canCapture(_ board: StoneLayout, target: Int) -> Bool {
            remaining = budget
            return attack(board, target, depth, root: true)
        }

        private mutating func nextStamp() -> UInt32 {
            stamp &+= 1
            if stamp == 0 {
                for i in mark.indices { mark[i] = 0 }
                stamp = 1
            }
            return stamp
        }

        /// Liberties (and, if asked, stones) of the chain at `p`, in depth-first discovery order.
        private mutating func chain(_ b: StoneLayout, _ p: Int, stones wantStones: Bool = false) -> (libs: [Int], stones: [Int]) {
            let s = nextStamp()
            let color = b[p]
            var libs: [Int] = []
            var stones: [Int] = []
            var stack = [p]
            mark[p] = s
            while let q = stack.popLast() {
                if wantStones { stones.append(q) }
                for k in 0 ..< 4 {
                    let r = g.adjacent[q * 4 + k]
                    guard r >= 0, mark[r] != s else { continue }
                    if b[r] == 0 {
                        mark[r] = s
                        libs.append(r)
                    } else if b[r] == color {
                        mark[r] = s
                        stack.append(r)
                    }
                }
            }
            return (libs, stones)
        }

        private mutating func libertyCount(_ b: StoneLayout, _ p: Int) -> Int { chain(b, p).libs.count }

        /// Plays `color` at `m`; nil if occupied or suicide.
        private mutating func play(_ b: StoneLayout, _ m: Int, _ color: UInt8) -> StoneLayout? {
            guard b[m] == 0 else { return nil }
            var t = b
            t[m] = color
            let enemy: UInt8 = color == 1 ? 2 : 1
            for k in 0 ..< 4 {
                let q = g.adjacent[m * 4 + k]
                guard q >= 0, t[q] == enemy else { continue }
                let captured = chain(t, q, stones: true)
                if captured.libs.isEmpty {
                    for s in captured.stones { t[s] = 0 }
                }
            }
            return libertyCount(t, m) == 0 ? nil : t
        }

        private func openness(_ b: StoneLayout, _ q: Int) -> Int {
            var n = 0
            for k in 0 ..< 4 {
                let r = g.adjacent[q * 4 + k]
                if r >= 0, b[r] == 0 { n += 1 }
            }
            return n
        }

        private mutating func attack(_ b: StoneLayout, _ p: Int, _ depth: Int, root: Bool) -> Bool {
            remaining -= 1
            if remaining < 0 { return false }
            let color = b[p]
            let libs = chain(b, p).libs
            if libs.count == 1 { return true }
            if depth <= 0 || libs.count > (root ? 3 : 2) { return false }
            let ordered = libs.sorted { a, c in
                let oa = openness(b, a), oc = openness(b, c)
                return oa != oc ? oa > oc : a < c
            }
            let attacker: UInt8 = color == 1 ? 2 : 1
            for m in ordered {
                guard let u = play(b, m, attacker) else { continue }
                if u[p] == 0 { return true }
                if !defend(u, p, depth - 1) { return true }
            }
            return false
        }

        /// Owner to move: true if the chain survives (an unfinished search counts as surviving).
        private mutating func defend(_ b: StoneLayout, _ p: Int, _ depth: Int) -> Bool {
            remaining -= 1
            if remaining < 0 { return true }
            let color = b[p]
            let (libs, stones) = chain(b, p, stones: true)
            if libs.count >= 3 || depth <= 0 { return true }
            var candidates = libs
            var seenChains = Set<Int>()
            for s in stones {
                for k in 0 ..< 4 {
                    let q = g.adjacent[s * 4 + k]
                    guard q >= 0, b[q] != 0, b[q] != color, !seenChains.contains(q) else { continue }
                    let enemy = chain(b, q, stones: true)
                    seenChains.formUnion(enemy.stones)
                    if enemy.libs.count <= 2 { candidates.append(contentsOf: enemy.libs) }
                }
            }
            var tried = Set<Int>()
            for m in candidates where tried.insert(m).inserted {
                guard let u = play(b, m, color) else { continue }
                if libertyCount(u, p) < 2 { continue }
                if !attack(u, p, depth - 1, root: false) { return true }
            }
            return false
        }
    }
}
