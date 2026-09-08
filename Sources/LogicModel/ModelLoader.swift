import Foundation

/// A single gate input reference: `(bank, channel, dx, dy)`.
/// bank 0 = previous layer output (layer 0: the 32 input channels); bank 1 = the 32 input
/// channels (layers ≥ 1 only). The input point is `(x+dx, y+dy)`; off-board reads 0.
public struct GateReference: Sendable, Equatable {
    public let bank: Int32
    public let channel: Int32
    public let dx: Int32
    public let dy: Int32
}

/// Fully validated in-memory model. Arrays are converted only after every structural,
/// size, and hash check succeeded.
public struct LogicModelData: Sendable {
    public let manifest: ModelManifest
    public let channels: Int
    public let layers: Int
    public let dilations: [Int]
    /// `[L][C][2]` references.
    public let wiring: [[[GateReference]]]
    /// `[L][C]` gate ids 0...15 (truth-table encoding: output = (g >> (2a+b)) & 1).
    public let gates: [[UInt8]]
    /// Head tensors by name, row-major float32 (`W` is `[in,out]`, `b` is `[out]`).
    public let heads: [String: [Float]]
    /// sha256 of the three payload files concatenated with their names (stable model identity).
    public let payloadHash: String

    public func head(_ name: String) -> [Float] { heads[name]! }
}

public enum ModelLoader {
    /// Loads and validates a `.ichigo` directory. Rejects symlinks, absolute/`..` names,
    /// unknown versions, size/hash mismatches, out-of-range wiring/gates, non-finite heads.
    public static func load(directory: URL) throws -> LogicModelData {
        let fm = FileManager.default
        let dirPath = directory.path
        if let attrs = try? fm.attributesOfItem(atPath: dirPath), attrs[.type] as? FileAttributeType == .typeSymbolicLink {
            throw LogicModelError.invalidPayload("model directory is a symlink")
        }
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dirPath, isDirectory: &isDir), isDir.boolValue else {
            throw LogicModelError.invalidPayload("\(dirPath) is not a directory")
        }
        let manifestData = try readRegularFile(directory, "manifest.json")
        let manifest = try ModelManifest.parse(data: manifestData)

        var payloads: [String: Data] = [:]
        for name in ModelManifest.fileNames {
            let entry = manifest.files[name]!
            let data = try readRegularFile(directory, name)
            guard data.count == entry.byteLength else {
                throw LogicModelError.invalidPayload("\(name): byteLength \(data.count) != manifest \(entry.byteLength)")
            }
            guard SHA256.hexDigest(data) == entry.sha256 else {
                throw LogicModelError.invalidPayload("\(name): sha256 mismatch")
            }
            payloads[name] = data
        }
        let L = manifest.layers
        let C = manifest.channels
        let wiringRaw = payloads["wiring.i32"]!.withUnsafeBytes { buf -> [Int32] in
            let n = buf.count / 4
            var out = [Int32](repeating: 0, count: n)
            for i in 0 ..< n { out[i] = Int32(littleEndian: buf.loadUnaligned(fromByteOffset: i * 4, as: Int32.self)) }
            return out
        }
        var wiring: [[[GateReference]]] = []
        wiring.reserveCapacity(L)
        for l in 0 ..< L {
            let d = Int32(manifest.dilations[l])
            var layer: [[GateReference]] = []
            layer.reserveCapacity(C)
            for c in 0 ..< C {
                var refs: [GateReference] = []
                for k in 0 ..< 2 {
                    let base = ((l * C + c) * 2 + k) * 4
                    let r = GateReference(bank: wiringRaw[base], channel: wiringRaw[base + 1], dx: wiringRaw[base + 2], dy: wiringRaw[base + 3])
                    guard r.bank == 0 || (r.bank == 1 && l > 0) else {
                        throw LogicModelError.invalidPayload("layer \(l) channel \(c): invalid bank \(r.bank)")
                    }
                    let bankChannels = (r.bank == 1 || l == 0) ? FeatureLayout.spatialChannels : C
                    guard r.channel >= 0, Int(r.channel) < bankChannels else {
                        throw LogicModelError.invalidPayload("layer \(l) channel \(c): channel \(r.channel) out of range")
                    }
                    guard [-d, 0, d].contains(r.dx), [-d, 0, d].contains(r.dy) else {
                        throw LogicModelError.invalidPayload("layer \(l) channel \(c): offset (\(r.dx),\(r.dy)) not in dilation \(d)")
                    }
                    refs.append(r)
                }
                guard refs[0] != refs[1] else { throw LogicModelError.invalidPayload("layer \(l) channel \(c): A == B") }
                layer.append(refs)
            }
            wiring.append(layer)
        }
        let gatesRaw = [UInt8](payloads["gates.u8"]!)
        var gates: [[UInt8]] = []
        for l in 0 ..< L {
            let row = Array(gatesRaw[(l * C) ..< ((l + 1) * C)])
            if let bad = row.first(where: { $0 > 15 }) {
                throw LogicModelError.invalidPayload("layer \(l): gate id \(bad) > 15")
            }
            gates.append(row)
        }
        let headsData = payloads["heads.f32"]!
        var heads: [String: [Float]] = [:]
        for t in manifest.headTensors {
            let count = t.byteLength / 4
            var values = [Float](repeating: 0, count: count)
            headsData.withUnsafeBytes { buf in
                for i in 0 ..< count {
                    values[i] = Float(bitPattern: UInt32(littleEndian: buf.loadUnaligned(fromByteOffset: t.byteOffset + i * 4, as: UInt32.self)))
                }
            }
            guard values.allSatisfy({ $0.isFinite }) else { throw LogicModelError.invalidPayload("head tensor \(t.name) contains non-finite values") }
            heads[t.name] = values
        }
        var hashInput = Data()
        for name in ModelManifest.fileNames {
            hashInput.append(name.data(using: .utf8)!)
            hashInput.append(payloads[name]!)
        }
        return LogicModelData(
            manifest: manifest, channels: C, layers: L, dilations: manifest.dilations, wiring: wiring, gates: gates,
            heads: heads, payloadHash: SHA256.hexDigest(hashInput)
        )
    }

    private static func readRegularFile(_ dir: URL, _ name: String) throws -> Data {
        guard !name.hasPrefix("/"), !name.contains(".."), !name.contains("/") else {
            throw LogicModelError.invalidPayload("illegal file name \(name)")
        }
        let url = dir.appendingPathComponent(name)
        let attrs: [FileAttributeKey: Any]
        do { attrs = try FileManager.default.attributesOfItem(atPath: url.path) } catch {
            throw LogicModelError.invalidPayload("\(name): missing")
        }
        guard attrs[.type] as? FileAttributeType == .typeRegular else {
            throw LogicModelError.invalidPayload("\(name): not a regular file (symlinks are rejected)")
        }
        do { return try Data(contentsOf: url) } catch {
            throw LogicModelError.invalidPayload("\(name): unreadable (\(error))")
        }
    }
}
