import Foundation
import LogicModel
import XCTest

/// Every corruption must be rejected before any array is built (T10 acceptance).
final class LoaderRejectionTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("ichigo-loader-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: FixturePaths.parity("tiny-9").appendingPathComponent("model.ichigo"), to: tmp)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    private func useFixture(_ name: String) throws {
        try FileManager.default.removeItem(at: tmp)
        try FileManager.default.copyItem(at: FixturePaths.parity(name).appendingPathComponent("model.ichigo"), to: tmp)
    }

    private func mutateManifest(_ f: (inout [String: Any]) -> Void) throws {
        let url = tmp.appendingPathComponent("manifest.json")
        var m = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        f(&m)
        try JSONSerialization.data(withJSONObject: m).write(to: url)
    }

    private func assertRejected(_ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try ModelLoader.load(directory: tmp), message, file: file, line: line)
    }

    func testValidLoads() throws {
        let m = try ModelLoader.load(directory: tmp)
        XCTAssertEqual(m.channels, 64)
        XCTAssertEqual(m.layers, 4)
        XCTAssertEqual(m.payloadHash.count, 64)
    }

    func testTruncatedPayload() throws {
        let url = tmp.appendingPathComponent("heads.f32")
        let d = try Data(contentsOf: url)
        try d.prefix(d.count - 4).write(to: url)
        assertRejected("truncated heads")
    }

    func testCorruptedByteHashMismatch() throws {
        let url = tmp.appendingPathComponent("gates.u8")
        var d = try Data(contentsOf: url)
        d[0] ^= 0x01
        try d.write(to: url)
        assertRejected("sha mismatch")
    }

    func testUnknownVersion() throws {
        try mutateManifest { $0["version"] = 2 }
        assertRejected("version")
    }

    func testUnknownFeatureVersionAndRules() throws {
        try mutateManifest { $0["featureVersion"] = 7 }
        assertRejected("featureVersion")
        try FileManager.default.removeItem(at: tmp)
        try FileManager.default.copyItem(at: FixturePaths.parity("tiny-9").appendingPathComponent("model.ichigo"), to: tmp)
        try mutateManifest { $0["rulesId"] = "japanese" }
        assertRejected("rulesId")
    }

    func testNaNInManifest() throws {
        let url = tmp.appendingPathComponent("manifest.json")
        let s = String(data: try Data(contentsOf: url), encoding: .utf8)!
        try s.replacingOccurrences(of: "\"calibrationTemperature\": 1.0", with: "\"calibrationTemperature\": NaN").write(to: url, atomically: true, encoding: .utf8)
        assertRejected("NaN")
    }

    func testGateOutOfRange() throws {
        let url = tmp.appendingPathComponent("gates.u8")
        var d = try Data(contentsOf: url)
        d[3] = 16
        try d.write(to: url)
        let sha = SHA256.hexDigest(d)
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            var g = files["gates.u8"] as! [String: Any]
            g["sha256"] = sha
            files["gates.u8"] = g
            m["files"] = files
        }
        assertRejected("gate 16")
    }

    func testWiringOutOfRange() throws {
        let url = tmp.appendingPathComponent("wiring.i32")
        var d = try Data(contentsOf: url)
        // layer 0, channel 0, ref B, field dx -> 5 (dilation is 1)
        let offset = (((0 * 64 + 0) * 2 + 1) * 4 + 2) * 4
        d[offset] = 5
        try d.write(to: url)
        let sha = SHA256.hexDigest(d)
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            var w = files["wiring.i32"] as! [String: Any]
            w["sha256"] = sha
            files["wiring.i32"] = w
            m["files"] = files
        }
        assertRejected("offset out of dilation")
    }

    func testHeadTensorProblems() throws {
        try mutateManifest { m in
            var t = m["headTensors"] as! [[String: Any]]
            t.removeLast()
            m["headTensors"] = t
        }
        assertRejected("missing tensor")
    }

    func testOverlapRejected() throws {
        try mutateManifest { m in
            var t = m["headTensors"] as! [[String: Any]]
            t[1]["byteOffset"] = t[0]["byteOffset"]
            m["headTensors"] = t
        }
        assertRejected("overlap")
    }

    func testExtraFileEntryAndBadChannels() throws {
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            files["../evil"] = ["byteLength": 0, "sha256": String(repeating: "0", count: 64)]
            m["files"] = files
        }
        assertRejected("path escape")
    }

    func testSymlinkRejected() throws {
        let url = tmp.appendingPathComponent("gates.u8")
        let real = tmp.appendingPathComponent("gates.real")
        try FileManager.default.moveItem(at: url, to: real)
        try FileManager.default.createSymbolicLink(at: url, withDestinationURL: real)
        assertRejected("symlink")
    }

    func testChannelRangeRejected() throws {
        try mutateManifest { $0["channels"] = 8 }
        assertRejected("channels < 16")
    }

    /// A huge byteLength must be rejected, not overflow the byte accounting.
    func testHugeByteLengthRejected() throws {
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            var h = files["heads.f32"] as! [String: Any]
            h["byteLength"] = 9_223_372_036_854_775_807
            files["heads.f32"] = h
            m["files"] = files
        }
        assertRejected("heads.f32 byteLength overflows")
        try FileManager.default.removeItem(at: tmp)
        try FileManager.default.copyItem(at: FixturePaths.parity("tiny-9").appendingPathComponent("model.ichigo"), to: tmp)
        try mutateManifest { m in
            var t = m["headTensors"] as! [[String: Any]]
            t[0]["byteLength"] = 9_223_372_036_854_775_807
            t[0]["byteOffset"] = 9_223_372_036_854_775_800
            m["headTensors"] = t
        }
        assertRejected("head tensor byteLength/byteOffset overflows")
    }

    /// 1e300 is a finite Double but +inf as a Float.
    func testCalibrationTemperatureNotFloatRepresentable() throws {
        try mutateManifest { $0["calibrationTemperature"] = 1e300 }
        assertRejected("calibrationTemperature overflows Float")
    }

    func testSHA256KnownVector() {
        XCTAssertEqual(SHA256.hexDigest(Data()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        XCTAssertEqual(SHA256.hexDigest("abc".data(using: .utf8)!), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testLUT4LoadsU16AndFourReferences() throws {
        try useFixture("tiny-9-lut4")
        let m = try ModelLoader.load(directory: tmp)
        XCTAssertEqual(m.manifest.gateArity, 4)
        XCTAssertEqual(m.manifest.gateEncoding, "lut4-msb-first")
        XCTAssertEqual(m.wiring[0][0].count, 4)
        XCTAssertEqual(m.gateTables[0].count, 64)
        XCTAssertGreaterThan(m.gateTables[0][0], 255)
    }

    func testLUT4RejectsU8GateManifest() throws {
        try useFixture("tiny-9-lut4")
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            let u16 = files.removeValue(forKey: "gates.u16")!
            files["gates.u8"] = u16
            m["files"] = files
        }
        assertRejected("arity 4 with gates.u8")
    }

    func testLUT4RejectsMismatchedWiringShape() throws {
        try useFixture("tiny-9-lut4")
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            var wiring = files["wiring.i32"] as! [String: Any]
            wiring["byteLength"] = (wiring["byteLength"] as! Int) - 16
            files["wiring.i32"] = wiring
            m["files"] = files
        }
        assertRejected("arity 4 mismatched wiring shape")
    }

    func testLUT4RejectsDuplicateReferences() throws {
        try useFixture("tiny-9-lut4")
        let url = tmp.appendingPathComponent("wiring.i32")
        var d = try Data(contentsOf: url)
        for i in 0 ..< 16 { d[16 + i] = d[i] }
        try d.write(to: url)
        let sha = SHA256.hexDigest(d)
        try mutateManifest { m in
            var files = m["files"] as! [String: Any]
            var wiring = files["wiring.i32"] as! [String: Any]
            wiring["sha256"] = sha
            files["wiring.i32"] = wiring
            m["files"] = files
        }
        assertRejected("arity 4 duplicate references")
    }
}
