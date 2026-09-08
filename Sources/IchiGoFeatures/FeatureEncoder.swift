import Foundation
import IchiGoCore

/// Encodes `PositionSnapshot`s into the v1 32-channel spatial + 4 global features
/// (docs/spec/01-network.md §1). Output layout `[B,S,S,32]` flat as `(((b*S+y)*S+x)*32+c)`.
public enum FeatureEncoder {
    public static let spatialChannels = 32
    public static let globalFeatures = 4
    public static let featureVersion = 1

    public struct Encoded: Sendable, Equatable {
        public let boardSize: Int
        public let batch: Int
        public let spatial: [UInt8]
        public let global: [Float]
        public let legal: [UInt8]
    }

    public enum EncodeError: Error, Equatable { case mixedBoardSizes }

    public static func encode(_ snapshots: [PositionSnapshot]) throws -> Encoded {
        guard let first = snapshots.first else { return Encoded(boardSize: 0, batch: 0, spatial: [], global: [], legal: []) }
        let S = first.boardSize
        guard snapshots.allSatisfy({ $0.boardSize == S }) else { throw EncodeError.mixedBoardSizes }
        var spatial = [UInt8](repeating: 0, count: snapshots.count * S * S * spatialChannels)
        var global = [Float](repeating: 0, count: snapshots.count * globalFeatures)
        var legal = [UInt8](repeating: 0, count: snapshots.count * (S * S + 1))
        for (b, snap) in snapshots.enumerated() {
            encodeOne(snap, into: &spatial, offset: b * S * S * spatialChannels)
            let g = globalFeatures(snap)
            for i in 0 ..< 4 { global[b * 4 + i] = g[i] }
            for i in 0 ..< (S * S + 1) { legal[b * (S * S + 1) + i] = snap.legal[i] }
        }
        return Encoded(boardSize: S, batch: snapshots.count, spatial: spatial, global: global, legal: legal)
    }

    /// `[4]`: komiSelf/(S*S), S/19, min(moveNumber,2S²)/(2S²), min(consecutivePasses,2)/2.
    public static func globalFeatures(_ snap: PositionSnapshot) -> [Float] {
        let S = Float(snap.boardSize)
        let komiSelf = snap.toMove == .white ? snap.komi : -snap.komi
        let twoArea = 2 * S * S
        return [
            komiSelf / (S * S),
            S / 19,
            Float(min(Float(snap.moveNumber), twoArea)) / twoArea,
            Float(min(snap.consecutivePasses, 2)) / 2,
        ]
    }

    static func encodeOne(_ snap: PositionSnapshot, into out: inout [UInt8], offset: Int) {
        let S = snap.boardSize
        let C = spatialChannels
        let me = UInt8(snap.toMove.rawValue)
        let opp = UInt8(snap.toMove.opponent.rawValue)
        @inline(__always) func set(_ x: Int, _ y: Int, _ c: Int) { out[offset + ((y * S + x) * C) + c] = 1 }
        let current = snap.layouts[0]
        for y in 0 ..< S {
            for x in 0 ..< S {
                let p = y * S + x
                // 0,1 current stones; 2..15 history t=1..7
                for t in 0 ..< 8 where t < snap.layouts.count {
                    let v = snap.layouts[t][p]
                    if v == me { set(x, y, 2 * t) } else if v == opp { set(x, y, 2 * t + 1) }
                }
                if current[p] == 0 { set(x, y, 16) }
                if snap.legal[p] == 1 { set(x, y, 17) }
                if snap.koPoint == p { set(x, y, 18) }
                let libs = snap.liberties[p]
                if current[p] == me || current[p] == opp {
                    let base = current[p] == me ? 19 : 22
                    if libs == 1 { set(x, y, base) } else if libs == 2 { set(x, y, base + 1) } else if libs >= 3 { set(x, y, base + 2) }
                }
                if x == 0 || y == 0 || x == S - 1 || y == S - 1 { set(x, y, 27) }
                set(x, y, 28)
            }
        }
        if let last = snap.recentMoves.first {
            switch last {
            case let .point(x, y): set(x, y, 25)
            case .pass: for p in 0 ..< (S * S) { set(p % S, p / S, 29) }
            }
        }
        if snap.recentMoves.count >= 2 {
            switch snap.recentMoves[1] {
            case let .point(x, y): set(x, y, 26)
            case .pass: for p in 0 ..< (S * S) { set(p % S, p / S, 30) }
            }
        }
        if snap.toMove == .black { for p in 0 ..< (S * S) { set(p % S, p / S, 31) } }
    }
}
