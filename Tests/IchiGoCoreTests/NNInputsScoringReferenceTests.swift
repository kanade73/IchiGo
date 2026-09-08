import Foundation
import IchiGoCore
import XCTest

final class NNInputsScoringReferenceTests: XCTestCase {
    func testGroupTaxScoring() throws {
        let board = try Board.parseBoard(13, 13, #"""
        .xo.o.o...xx.
        xoooooo...x.x
        oo.........xx
        ..xx.........
        xx..x........
        .x..x........
        ox..x........
        .xx.x........
        .x.x.........
        xxxx........x
        oooxooo...xx.
        o.oxo.o...x.x
        .o.o.o....xx.
        """#)
        let area = board.calculateArea(
            nonPassAliveStones: true,
            safeBigTerritories: true,
            unsafeBigTerritories: true,
            isMultiStoneSuicideLegal: false
        )
        let expected = #"""
        No group tax
         100  100  100  100  100  100  100    0    0    0 -100 -100 -100
         100  100  100  100  100  100  100    0    0    0 -100 -100 -100
         100  100    0    0    0    0    0    0    0    0    0 -100 -100
           0    0 -100 -100    0    0    0    0    0    0    0    0    0
        -100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
        -100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
        -100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
        -100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
        -100 -100 -100 -100    0    0    0    0    0    0    0    0    0
        -100 -100 -100 -100    0    0    0    0    0    0    0    0 -100
         100  100  100 -100  100  100  100    0    0    0 -100 -100 -100
         100  100  100 -100  100  100  100    0    0    0 -100 -100 -100
         100  100  100  100  100  100    0    0    0    0 -100 -100 -100

        Group tax
          60   60  100   60  100   60  100    0    0    0 -100 -100    0
          60  100  100  100  100  100  100    0    0    0 -100    0 -100
         100  100    0    0    0    0    0    0    0    0    0 -100 -100
           0    0 -100 -100    0    0    0    0    0    0    0    0    0
        -100 -100  -83  -83 -100    0    0    0    0    0    0    0    0
         -83 -100  -83  -83 -100    0    0    0    0    0    0    0    0
         -83 -100  -83  -83 -100    0    0    0    0    0    0    0    0
         -83 -100 -100  -83 -100    0    0    0    0    0    0    0    0
         -83 -100  -83 -100    0    0    0    0    0    0    0    0    0
        -100 -100 -100 -100    0    0    0    0    0    0    0    0 -100
         100  100  100 -100  100  100  100    0    0    0 -100 -100  -33
         100   60  100 -100  100   60  100    0    0    0 -100  -33 -100
          60  100   60  100   60  100    0    0    0    0 -100 -100  -33
        """#
        XCTAssertEqual(render(board, area), expected)
    }

    func testGroupTaxScoring2() throws {
        let board = try board2()
        let area = board.calculateArea(
            nonPassAliveStones: true,
            safeBigTerritories: true,
            unsafeBigTerritories: false,
            isMultiStoneSuicideLegal: true
        )
        let grid = #"""
           0 -100  100    0  100 -100    0  100  100  100  100  100  100
        -100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
         100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
        -100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
         100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
           0  100    0  100 -100    0    0  100  100  100  100  100  100
           0  100  100 -100 -100    0    0    0    0    0    0    0    0
         100  100 -100 -100    0    0    0    0    0    0    0    0    0
           0    0    0    0    0    0    0    0    0    0    0    0    0
         100  100  100 -100 -100    0    0    0    0  100    0    0    0
         100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
         100 -100    0  100 -100  100  100  100  100 -100  100  100  100
         100 -100    0  100 -100  100    0  100    0 -100    0  100    0
        """#
        let expected = "No group tax\n\(grid)\n\nGroup tax\n\(grid)"
        XCTAssertEqual(render(board, area), expected)
    }

    func testGroupTaxScoring2WithUnsafeTerritories() throws {
        let board = try board2()
        let area = board.calculateArea(
            nonPassAliveStones: true,
            safeBigTerritories: true,
            unsafeBigTerritories: true,
            isMultiStoneSuicideLegal: true
        )
        let expected = #"""
        No group tax
        -100 -100  100  100  100 -100    0  100  100  100  100  100  100
        -100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
         100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
        -100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
         100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
         100  100  100  100 -100    0    0  100  100  100  100  100  100
         100  100  100 -100 -100    0    0    0    0    0    0    0    0
         100  100 -100 -100    0    0    0    0    0    0    0    0    0
           0    0    0    0    0    0    0    0    0    0    0    0    0
         100  100  100 -100 -100    0    0    0    0  100    0    0    0
         100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
         100 -100    0  100 -100  100  100  100  100 -100  100  100  100
         100 -100    0  100 -100  100  100  100    0 -100    0  100  100

        Group tax
           0 -100  100    0  100 -100    0  100  100  100  100  100  100
        -100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
         100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
        -100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
         100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
          33  100   33  100 -100    0    0  100  100  100  100  100  100
          33  100  100 -100 -100    0    0    0    0    0    0    0    0
         100  100 -100 -100    0    0    0    0    0    0    0    0    0
           0    0    0    0    0    0    0    0    0    0    0    0    0
         100  100  100 -100 -100    0    0    0    0  100    0    0    0
         100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
         100 -100    0  100 -100  100  100  100  100 -100  100  100  100
         100 -100    0  100 -100  100    0  100    0 -100    0  100    0
        """#
        XCTAssertEqual(render(board, area), expected)
    }

    private func board2() throws -> Board {
        try Board.parseBoard(13, 13, #"""
        .xo.ox.oooooo
        x.ooox.oxxxxo
        oooxxx.oxo..x
        xxxx...oxo..x
        oooxx..oxxxxo
        .o.ox..oooooo
        .ooxx........
        ooxx.........
        .............
        oooxx....o...
        oxxoxxxxxoxxx
        ox.oxooooxooo
        ox.oxo.o.x.o.
        """#)
    }

    private func render(_ board: Board, _ area: [Color]) -> String {
        func grid(_ groupTax: Bool) -> String {
            let scoring = NNInputs.fillScoring(board, area: area, groupTax: groupTax)
            return (0 ..< board.ySize).map { y in
                (0 ..< board.xSize).map { x in
                    String(format: "%4.0f", scoring[board.getLoc(x, y)] * 100)
                }.joined(separator: " ")
            }.joined(separator: "\n")
        }
        return "No group tax\n\(grid(false))\n\nGroup tax\n\(grid(true))"
    }
}
