import IchiGoCore
import IchiGoFeatures
import XCTest

final class CoordinateTests: XCTestCase {
    func testGTPCorners() throws {
        XCTAssertEqual(try Coordinates.parseGTP("A9", size: 9), .point(x: 0, y: 0))
        XCTAssertEqual(Coordinates.index(of: try Coordinates.parseGTP("A9", size: 9), size: 9), 0)
        XCTAssertEqual(try Coordinates.parseGTP("J1", size: 9), .point(x: 8, y: 8))
        XCTAssertEqual(Coordinates.index(of: try Coordinates.parseGTP("J1", size: 9), size: 9), 80)
        XCTAssertEqual(Coordinates.index(of: try Coordinates.parseGTP("A19", size: 19), size: 19), 0)
        XCTAssertEqual(Coordinates.index(of: try Coordinates.parseGTP("T1", size: 19), size: 19), 360)
        XCTAssertEqual(Coordinates.gtpString(.point(x: 0, y: 0), size: 9), "A9")
        XCTAssertEqual(Coordinates.gtpString(.point(x: 8, y: 8), size: 9), "J1")
        XCTAssertEqual(Coordinates.gtpString(.point(x: 18, y: 18), size: 19), "T1")
        XCTAssertEqual(try Coordinates.parseGTP("pass", size: 9), .pass)
        XCTAssertEqual(Coordinates.index(of: .pass, size: 9), 81)
    }

    func testIColumnRejected() {
        XCTAssertThrowsError(try Coordinates.parseGTP("I5", size: 9))
        XCTAssertThrowsError(try Coordinates.parseGTP("K1", size: 9))   // beyond 9 columns
        XCTAssertThrowsError(try Coordinates.parseGTP("A10", size: 9))
        XCTAssertThrowsError(try Coordinates.parseGTP("A0", size: 9))
    }

    func testSGF() throws {
        XCTAssertEqual(try Coordinates.parseSGF("aa", size: 9), .point(x: 0, y: 0))
        XCTAssertEqual(try Coordinates.parseSGF("ca", size: 9), .point(x: 2, y: 0))
        XCTAssertEqual(try Coordinates.parseSGF("", size: 9), .pass)
        XCTAssertEqual(try Coordinates.parseSGF("tt", size: 19), .pass)
        XCTAssertThrowsError(try Coordinates.parseSGF("ja", size: 9))
        XCTAssertEqual(Coordinates.sgfString(.point(x: 2, y: 5), size: 9), "cf")
    }

    func testLocRoundTrip() {
        for s in [9, 19] {
            for y in 0 ..< s {
                for x in 0 ..< s {
                    let m = MoveCoord.point(x: x, y: y)
                    XCTAssertEqual(Coordinates.move(fromLoc: Coordinates.loc(m, size: s), size: s), m)
                }
            }
            XCTAssertEqual(Coordinates.move(fromLoc: Coordinates.loc(.pass, size: s), size: s), .pass)
        }
    }
}
