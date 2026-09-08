import Foundation

/// D4 symmetries (docs/spec/01-network.md §1). `R(x,y)=(S-1-y,x)`, `F(x,y)=(S-1-x,y)`;
/// id 0..3 = R^id, id 4..7 = R^(id-4)∘F. `forward[p]` is where point `p` moves; transformed
/// planes satisfy `T[forward[p]] = X[p]`. Pass and global are unchanged. The inverse table is
/// built independently from the forward table (not by picking another id).
public struct FeatureSymmetry: Sendable {
    public static let count = 8
    public let boardSize: Int
    public let forward: [[Int]]
    public let inverse: [[Int]]

    public init(boardSize: Int) {
        self.boardSize = boardSize
        var fwd: [[Int]] = []
        var inv: [[Int]] = []
        for sym in 0 ..< Self.count {
            var f = [Int](repeating: 0, count: boardSize * boardSize)
            for y in 0 ..< boardSize {
                for x in 0 ..< boardSize {
                    let (nx, ny) = Self.map(x: x, y: y, size: boardSize, sym: sym)
                    f[y * boardSize + x] = ny * boardSize + nx
                }
            }
            var i = [Int](repeating: 0, count: f.count)
            for (p, q) in f.enumerated() { i[q] = p }
            fwd.append(f)
            inv.append(i)
        }
        forward = fwd
        inverse = inv
    }

    public static func map(x: Int, y: Int, size: Int, sym: Int) -> (Int, Int) {
        precondition(sym >= 0 && sym < count)
        var (px, py) = (x, y)
        if sym >= 4 { px = size - 1 - px }
        for _ in 0 ..< (sym % 4) { (px, py) = (size - 1 - py, px) }
        return (px, py)
    }

    /// `[B,S,S,C]` flat, every channel plane permuted.
    public func transformSpatial(_ spatial: [UInt8], batch: Int, channels: Int, sym: Int, inverse useInverse: Bool = false) -> [UInt8] {
        let P = boardSize * boardSize
        let perm = useInverse ? inverse[sym] : forward[sym]
        var out = [UInt8](repeating: 0, count: spatial.count)
        for b in 0 ..< batch {
            for p in 0 ..< P {
                let src = (b * P + p) * channels
                let dst = (b * P + perm[p]) * channels
                for c in 0 ..< channels { out[dst + c] = spatial[src + c] }
            }
        }
        return out
    }

    /// `[S*S+1]`: board entries permuted, pass kept.
    public func transformPolicy<T>(_ policy: [T], sym: Int, inverse useInverse: Bool = false) -> [T] {
        let P = boardSize * boardSize
        precondition(policy.count == P + 1)
        let perm = useInverse ? inverse[sym] : forward[sym]
        var out = policy
        for p in 0 ..< P { out[perm[p]] = policy[p] }
        out[P] = policy[P]
        return out
    }

    /// `[S*S]` ownership or any per-point plane.
    public func transformPlane<T>(_ plane: [T], sym: Int, inverse useInverse: Bool = false) -> [T] {
        let P = boardSize * boardSize
        precondition(plane.count == P)
        let perm = useInverse ? inverse[sym] : forward[sym]
        var out = plane
        for p in 0 ..< P { out[perm[p]] = plane[p] }
        return out
    }
}
