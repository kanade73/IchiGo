/// Deterministic, seedable RNG (SplitMix64) used by `ValueSource.rollout` to sample moves from a
/// leaf's policy during playouts (docs/spec/03-engine.md §3-4 "値ソース"). Swift's
/// `SystemRandomNumberGenerator` cannot be seeded, so playout move sampling needs its own
/// generator for the whole search's rollout sequence to be reproducible from `SearchSettings`
/// alone, given a deterministic evaluator. Not used by any other `ValueSource` mode.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
