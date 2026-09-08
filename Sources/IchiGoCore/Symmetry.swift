/// Dihedral (D4) board symmetries, ported bit-for-bit from KataGo's `SymmetryHelpers`
/// (`cpp/neuralnet/nninputs.cpp`). There are 8 symmetries indexed `0...7`; the three low bits
/// select, in KataGo's fixed order, a horizontal flip (`0x2`), a vertical flip (`0x1`), and a
/// transpose (`0x4`), applied flip-then-transpose. These pure functions are the coordinate algebra
/// the opening-book converter (`bookconvert`) uses to compose the per-link symmetry transforms the
/// book stores (`linkSyms`) onto our own concrete board coordinates.
///
/// Conventions (must match KataGo exactly, since the book was built by KataGo):
/// - `apply(x:y:size:_:)` is `SymmetryHelpers::getSymLoc` for a square board: the point `(x, y)`
///   in the *source* frame lands at the returned `(x, y)` in the *destination* frame. Equivalently,
///   the destination board `D` relates to the source board `S` by `D[apply(p, s)] = S[p]` — i.e.
///   applying `s` to a board moves the stone at `p` to `apply(p, s)`.
/// - `compose(_:_:)` returns the single symmetry equal to "apply `first`, then `next`", so
///   `apply(apply(p, first), next) == apply(p, compose(first, next))` for every point.
/// - `invert(_:)` returns the group inverse: `compose(s, invert(s)) == 0` (identity) for all `s`.
public enum Symmetry {
    /// Number of dihedral symmetries of a square board.
    public static let count = 8

    /// True iff the symmetry includes a transpose (swaps the x and y axes).
    @inlinable
    public static func isTranspose(_ symmetry: Int) -> Bool {
        (symmetry & 0x4) != 0
    }

    /// Group inverse. Only symmetries 5 and 6 (the two "rotate 90°" elements) are non-involutions
    /// and invert to each other; every other symmetry is its own inverse.
    @inlinable
    public static func invert(_ symmetry: Int) -> Int {
        if symmetry == 5 { return 6 }
        if symmetry == 6 { return 5 }
        return symmetry
    }

    /// Compose two symmetries into one that is equivalent to applying `first` and then `next`.
    /// Mirrors `SymmetryHelpers::compose`: a leading transpose swaps the flip bits of the following
    /// symmetry before the XOR, because a transpose exchanges the roles of the x- and y-flips.
    @inlinable
    public static func compose(_ first: Int, _ next: Int) -> Int {
        var next = next
        if isTranspose(first) {
            next = (next & 0x4) | ((next & 0x2) >> 1) | ((next & 0x1) << 1)
        }
        return first ^ next
    }

    /// Three-way compose: apply `first`, then `next`, then `nextNext`.
    @inlinable
    public static func compose(_ first: Int, _ next: Int, _ nextNext: Int) -> Int {
        compose(compose(first, next), nextNext)
    }

    /// Map a point `(x, y)` on a square board of side `size` through `symmetry`.
    /// Flip x (`0x2`), then flip y (`0x1`), then transpose (`0x4`) — KataGo's exact order.
    @inlinable
    public static func apply(x: Int, y: Int, size: Int, _ symmetry: Int) -> (x: Int, y: Int) {
        var x = x
        var y = y
        if (symmetry & 0x2) != 0 { x = size - 1 - x }
        if (symmetry & 0x1) != 0 { y = size - 1 - y }
        if (symmetry & 0x4) != 0 { swap(&x, &y) }
        return (x, y)
    }
}
