import Foundation

/// Parsed and validated `manifest.json` of a `.ichigo` v1 directory (docs/spec/01-network.md §6).
/// Only the fixed v1 values are accepted; anything else is rejected before any payload is read.
public struct ModelManifest: Sendable, Equatable {
    public struct FileEntry: Sendable, Equatable {
        public let byteLength: Int
        public let sha256: String
    }

    public struct HeadTensor: Sendable, Equatable {
        public let name: String
        public let shape: [Int]
        public let byteOffset: Int
        public let byteLength: Int
    }

    public static let format = "ichigo.logic"
    public static let version = 1
    public static let featureVersion = 1
    public static let supportedHeadVersions: Set<Int> = [1, 2]
    public static let rulesID = "cgos-area-psk-v1"
    public static let gateEncoding = "truth-table-lsb-2a-plus-b"
    public static let fileNames = ["wiring.i32", "gates.u8", "heads.f32"]
    public static let supportedBoardSizes: Set<Int> = [9, 19]
    public static let channelRange = 16 ... 4096
    public static let layerRange = 1 ... 64
    public static let dilationRange = 1 ... 19
    public static let maxPayloadBytes = 512 * 1024 * 1024
    public static let headTensorNames = [
        "Wlocal", "blocal", "Wpolicy", "bpolicy", "Wowner", "bowner",
        "Wglobal", "bglobal", "Wpass", "bpass", "Wwdl", "bwdl", "Wscore", "bscore",
    ]
    public static let localHidden = 64
    public static let globalHidden = 128

    public let headVersion: Int
    public let boardSizes: [Int]
    public let channels: Int
    public let layers: Int
    public let dilations: [Int]
    public let calibrationTemperature: Float
    public let files: [String: FileEntry]
    public let headTensors: [HeadTensor]
    /// `trainingProvenance` re-serialised as canonical JSON text (kept as text so the struct stays Sendable).
    public let trainingProvenanceJSON: String
    /// Original manifest JSON text (for `inspect`).
    public let rawJSON: String

    public static func == (a: ModelManifest, b: ModelManifest) -> Bool {
        a.boardSizes == b.boardSizes && a.channels == b.channels && a.dilations == b.dilations
            && a.files == b.files && a.headTensors == b.headTensors
    }

    /// headVersion 1: u_global = concat(m, v, global) → 2C+4.
    /// headVersion 2: u_global = concat(m, v, mean_xy(z_xy)[64], mean_xy(ownership)[1], global) → 2C+69.
    public static func globalInputSize(channels C: Int, headVersion: Int) -> Int {
        headVersion == 1 ? 2 * C + 4 : 2 * C + localHidden + 1 + 4
    }

    public static func headShapes(channels C: Int, headVersion: Int) -> [String: [Int]] {
        let G = globalInputSize(channels: C, headVersion: headVersion)
        return [
            "Wlocal": [3 * C + 4, localHidden], "blocal": [localHidden],
            "Wpolicy": [localHidden, 1], "bpolicy": [1],
            "Wowner": [localHidden, 1], "bowner": [1],
            "Wglobal": [G, globalHidden], "bglobal": [globalHidden],
            "Wpass": [globalHidden, 1], "bpass": [1],
            "Wwdl": [globalHidden, 3], "bwdl": [3],
            "Wscore": [globalHidden, 1], "bscore": [1],
        ]
    }

    /// Parses and validates. `JSONSerialization` already rejects NaN/Infinity tokens.
    public static func parse(data: Data) throws -> ModelManifest {
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw LogicModelError.invalidManifest("not valid JSON (\(error))")
        }
        guard let dict = object as? [String: Any] else { throw LogicModelError.invalidManifest("top level is not an object") }
        return try validate(dict)
    }

    // swiftlint:disable:next cyclomatic_complexity function_body_length
    static func validate(_ m: [String: Any]) throws -> ModelManifest {
        func str(_ k: String) throws -> String {
            guard let v = m[k] as? String else { throw LogicModelError.invalidManifest("missing or non-string \(k)") }
            return v
        }
        func int(_ k: String) throws -> Int {
            guard let n = m[k] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                  n.doubleValue == n.doubleValue.rounded(), let v = m[k] as? Int else {
                throw LogicModelError.invalidManifest("missing or non-integer \(k)")
            }
            return v
        }
        let required = ["format", "version", "featureVersion", "headVersion", "boardSizes", "rulesId", "channels", "layers",
                        "dilations", "gateEncoding", "layout", "endianness", "valuePerspective", "wdlOrder",
                        "calibrationTemperature", "files", "headTensors", "trainingProvenance"]
        for k in required where m[k] == nil { throw LogicModelError.invalidManifest("missing key \(k)") }
        guard try str("format") == format else { throw LogicModelError.invalidManifest("unknown format") }
        guard try int("version") == version else { throw LogicModelError.invalidManifest("unsupported version \(m["version"] ?? "")") }
        guard try int("featureVersion") == featureVersion else { throw LogicModelError.invalidManifest("unsupported featureVersion") }
        let headVersion = try int("headVersion")
        guard supportedHeadVersions.contains(headVersion) else { throw LogicModelError.invalidManifest("unsupported headVersion \(headVersion)") }
        guard try str("rulesId") == rulesID else { throw LogicModelError.invalidManifest("unsupported rulesId") }
        guard try str("gateEncoding") == gateEncoding else { throw LogicModelError.invalidManifest("unsupported gateEncoding") }
        guard try str("layout") == "NHWC" else { throw LogicModelError.invalidManifest("unsupported layout") }
        guard try str("endianness") == "little" else { throw LogicModelError.invalidManifest("unsupported endianness") }
        guard try str("valuePerspective") == "to-move" else { throw LogicModelError.invalidManifest("unsupported valuePerspective") }
        guard let wdl = m["wdlOrder"] as? [String], wdl == ["win", "draw", "loss"] else {
            throw LogicModelError.invalidManifest("unsupported wdlOrder")
        }
        guard let temp = m["calibrationTemperature"] as? NSNumber, temp.doubleValue > 0, temp.doubleValue.isFinite,
              Float(temp.doubleValue).isFinite, Float(temp.doubleValue) > 0 else {
            throw LogicModelError.invalidManifest("calibrationTemperature must be positive and representable as a finite Float")
        }
        guard let sizesAny = m["boardSizes"] as? [Int], !sizesAny.isEmpty, sizesAny.allSatisfy({ supportedBoardSizes.contains($0) }) else {
            throw LogicModelError.invalidManifest("boardSizes must be a non-empty subset of \(supportedBoardSizes.sorted())")
        }
        let channels = try int("channels")
        let layers = try int("layers")
        guard let dilations = m["dilations"] as? [Int] else { throw LogicModelError.invalidManifest("dilations must be an int array") }
        guard channelRange.contains(channels) else { throw LogicModelError.invalidManifest("channels \(channels) outside \(channelRange)") }
        guard layerRange.contains(layers) else { throw LogicModelError.invalidManifest("layers \(layers) outside \(layerRange)") }
        guard dilations.count == layers else { throw LogicModelError.invalidManifest("dilations count != layers") }
        for d in dilations where !dilationRange.contains(d) { throw LogicModelError.invalidManifest("dilation \(d) outside \(dilationRange)") }

        // Byte sizes with overflow checks before any multiplication is trusted.
        let (lc, o1) = layers.multipliedReportingOverflow(by: channels)
        let (wiringLen, o2) = lc.multipliedReportingOverflow(by: 32)
        guard !o1, !o2, wiringLen + lc <= maxPayloadBytes else { throw LogicModelError.invalidManifest("payload too large") }
        let gatesLen = lc

        guard let filesAny = m["files"] as? [String: Any], Set(filesAny.keys) == Set(fileNames) else {
            throw LogicModelError.invalidManifest("files must list exactly \(fileNames)")
        }
        var files: [String: FileEntry] = [:]
        for name in fileNames {
            guard let e = filesAny[name] as? [String: Any], let len = e["byteLength"] as? Int, len >= 0, len <= maxPayloadBytes,
                  let sha = e["sha256"] as? String, sha.count == 64, sha.allSatisfy({ "0123456789abcdef".contains($0) }) else {
                throw LogicModelError.invalidManifest("files.\(name) invalid")
            }
            files[name] = FileEntry(byteLength: len, sha256: sha)
        }
        guard files["wiring.i32"]!.byteLength == wiringLen else { throw LogicModelError.invalidManifest("wiring.i32 byteLength != L*C*2*4*4") }
        guard files["gates.u8"]!.byteLength == gatesLen else { throw LogicModelError.invalidManifest("gates.u8 byteLength != L*C") }
        let headsLen = files["heads.f32"]!.byteLength
        let (wg, o4) = wiringLen.addingReportingOverflow(gatesLen)
        let (payload, o5) = wg.addingReportingOverflow(headsLen)
        guard !o4, !o5, payload <= maxPayloadBytes else { throw LogicModelError.invalidManifest("payload too large") }

        guard let entriesAny = m["headTensors"] as? [[String: Any]] else { throw LogicModelError.invalidManifest("headTensors must be an array") }
        let shapes = headShapes(channels: channels, headVersion: headVersion)
        var seen = Set<String>()
        var tensors: [HeadTensor] = []
        var total = 0
        for e in entriesAny {
            guard let name = e["name"] as? String, let expected = shapes[name] else {
                throw LogicModelError.invalidManifest("unknown head tensor \(e["name"] ?? "?")")
            }
            guard seen.insert(name).inserted else { throw LogicModelError.invalidManifest("duplicate head tensor \(name)") }
            guard let shape = e["shape"] as? [Int], shape == expected else {
                throw LogicModelError.invalidManifest("head tensor \(name) shape mismatch, expected \(expected)")
            }
            guard let off = e["byteOffset"] as? Int, let len = e["byteLength"] as? Int,
                  off >= 0, len >= 0, off <= maxPayloadBytes, len <= maxPayloadBytes else {
                throw LogicModelError.invalidManifest("head tensor \(name): invalid offset/length")
            }
            guard off % 4 == 0 else { throw LogicModelError.invalidManifest("head tensor \(name): byteOffset not 4-byte aligned") }
            let count = expected.reduce(1, *)
            guard len == count * 4 else { throw LogicModelError.invalidManifest("head tensor \(name): byteLength != 4*prod(shape)") }
            let (end, o3) = off.addingReportingOverflow(len)
            guard !o3, end <= headsLen else { throw LogicModelError.invalidManifest("head tensor \(name) exceeds heads.f32") }
            tensors.append(HeadTensor(name: name, shape: shape, byteOffset: off, byteLength: len))
            let (sum, o6) = total.addingReportingOverflow(len)
            guard !o6, sum <= maxPayloadBytes else { throw LogicModelError.invalidManifest("payload too large") }
            total = sum
        }
        guard seen == Set(headTensorNames) else {
            throw LogicModelError.invalidManifest("missing head tensors \(Set(headTensorNames).subtracting(seen).sorted())")
        }
        let sorted = tensors.sorted { $0.byteOffset < $1.byteOffset }
        for (a, b) in zip(sorted, sorted.dropFirst()) where b.byteOffset < a.byteOffset + a.byteLength {
            throw LogicModelError.invalidManifest("head tensors \(a.name) and \(b.name) overlap")
        }
        guard total == headsLen else { throw LogicModelError.invalidManifest("heads.f32 has bytes not covered by headTensors") }
        guard let prov = m["trainingProvenance"] as? [String: Any] else {
            throw LogicModelError.invalidManifest("trainingProvenance must be an object")
        }
        let provJSON = String(data: try JSONSerialization.data(withJSONObject: prov, options: [.sortedKeys]), encoding: .utf8) ?? "{}"
        let rawJSON = String(data: try JSONSerialization.data(withJSONObject: m, options: [.sortedKeys]), encoding: .utf8) ?? "{}"
        return ModelManifest(
            headVersion: headVersion, boardSizes: sizesAny, channels: channels, layers: layers, dilations: dilations,
            calibrationTemperature: Float(temp.doubleValue), files: files, headTensors: tensors,
            trainingProvenanceJSON: provJSON, rawJSON: rawJSON
        )
    }
}
