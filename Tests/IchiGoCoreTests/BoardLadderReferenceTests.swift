import IchiGoCore
import XCTest

final class BoardLadderReferenceTests: XCTestCase {
    func testOneLib() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        xo.x..oxo
        xoxo..o..
        xxo......
        ..o.x....
        xo..xox..
        o..ooxo..
        .....xo..
        xoox..xo.
        .xxoo.xxo

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: true)
        actual += "\n"
        let expected = #"""

        01.0..010
        0100..0..
        000......
        ..0.0....
        10..000..
        0..0000..
        .....00..
        0000..00.
        .1100.001

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testTwoLibs() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        xo.x..oxo
        xo.o..o..
        xxo......
        ..o.x....
        xo..xo...
        ...ooxo..
        .....xo..
        xoox..xo.
        .xx.o.xxo

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        11.1..000
        11.0..0..
        110......
        ..0.0....
        10..00...
        ...0010..
        .....10..
        1110..01.
        .11.0.000


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testKo1() throws {
        let board1 = try Board.parseBoard(18, 9, #"""

        ..................
        ..................
        ....ox.......ox...
        ..xooox....xooox..
        ..xoxox....xoxox..
        ..xx.x......x.x...
        ...ox.......ox....
        ....o.............
        ..................

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        ..................
        ..................
        ....10.......00...
        ..01110....00000..
        ..01010....00000..
        ..00.0......0.0...
        ...00.......00....
        ....0.............
        ..................


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testKo2() throws {
        let board1 = try Board.parseBoard(18, 9, #"""

        .............xo.oo
        ....x........xxooo
        ...x.xx.......xxo.
        ..xoxoxo.......xxo
        ..xoooxo........xx
        ...xo......o......
        ..................
        ..................
        .....x.oo.........

        """#)
        let board2 = try Board.parseBoard(18, 9, #"""

        ..................
        ....x.............
        ...x.xx...........
        ..xoxoxo..........
        ..xoooxo..........
        ...xo......o......
        .................x
        ..................
        .....x.oo.........

        """#)
        let board3 = try Board.parseBoard(18, 9, #"""

        ....xo.......xo...
        ....xox......xox..
        ...xo.ox....xo.ox.
        ...xxoox....xxoox.
        ..xooox....xooox..
        .xo.ox....xo.ox...
        ..xox......xox....
        ..xox......xox....
        ............o.....

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board2, defenderFirst: false)
        actual += "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board3, defenderFirst: false)
        actual += "\n"
        let expected = #"""


        .............00.11
        ....0........00111
        ...0.00.......001.
        ..000000.......000
        ..000000........00
        ...00......0......
        ..................
        ..................
        .....0.00.........

        ..................
        ....0.............
        ...0.00...........
        ..010100..........
        ..011100..........
        ...01......0......
        .................0
        ..................
        .....0.00.........

        ....01.......01...
        ....011......011..
        ...00.10....00.00.
        ...00110....00000.
        ..01110....00000..
        .00.10....00.00...
        ..010......000....
        ..010......000....
        ............0.....


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testMultiKo() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        .xxxxxxo.
        xoxox.xo.
        ooo.oxoo.
        ..oooo...
        .xxxx....
        xx.x.xx..
        xoxoxox..
        xoooooxx.
        o.ooo.ox.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        .0000000.
        00000.00.
        000.0000.
        ..0000...
        .0000....
        00.0.00..
        0000000..
        00000000.
        0.000.00.

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testBigBoard() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

           A B C D E F G H J K L M N O P Q R S T
        19 . . . . . . . . . . . . . . . . . . .
        18 . . . . . . . . . . . . . . . . . . .
        17 . . . . . . . . . . . . . . . . . . .
        16 . . . X . . . . . . . . . . . X . . .
        15 . . . . . . . . . . . . . . . . . . .
        14 . . . . . . . . . . . . . . . . . . .
        13 . . . . . . . . . . . . . . . . . . .
        12 . . . . . . . . . . . . . . . . . . .
        11 . . . . . . . . . . . . . . . . . . .
        10 . . . . . . . . . . . . . . . . . . .
         9 . . . . . . . . . . . . . . . . . . .
         8 . . . . . . . . . . . . . . . . . . .
         7 . . . . . . . . . . . . . . . . . . .
         6 . . . . . . . . . . . . . . . . . . .
         5 . . . . . . . . . . . . . . . . . . .
         4 . . . . . . . . . . . . . . X O O . .
         3 . . . O . . . . . . . . . . O X X . .
         2 . . . . . . . . . . . . . . . . . . .
         1 . . . . . . . . . . . . . . . . . . .

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        ...................
        ...................
        ...................
        ...0...........0...
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ..............000..
        ...0..........000..
        ...................
        ...................

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testWholeBoard() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

           A B C D E F G H J K L M N O P Q R S T
        19 . . . . O . . . . . . . . . . . . . .
        18 O . . . . . . . . . . . . . . . . . O
        17 . . . . . . . . . . . . . . . . . . O
        16 . . . O O X O . . . . . . . . . . . .
        15 O . . . O X O O O O . O . . . . . O .
        14 O X X X X X X X X X O . . . . . . . .
        13 X O O O . X . . . . X . . . . . . . O
        12 X . O O O X . . . . X O . . . . . . .
        11 X O O O O X . . . . X . . . . . . . .
        10 O X X X X X X X X X O . . . . . . . O
         9 O O . . . X O . . O . . . . . . . . .
         8 . O . O . X . O . . . . . . . . . . .
         7 . . . . . . O . . . . . . . . . . . .
         6 . . . . O . . . . X . . . . . . . . .
         5 O . . . . X . . . X . . . X O . . . .
         4 . . . . . X . . . X . . . X . . . . .
         3 . . . . . X . . . X . . . X . . . . .
         2 . . . . O . X X X X X X X O . . . O .
         1 . O . . . . O O O O O O O . O . . . .

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        ....0..............
        0.................0
        ..................0
        ...0000............
        0...000000.0.....0.
        00000000000........
        0000.0....0.......0
        0.0000....00.......
        000000....0........
        00000000000.......0
        00...00..0.........
        .0.0.0.0...........
        ......0............
        ....0....0.........
        0....0...0...00....
        .....0...0...0.....
        .....0...0...0.....
        ....0.00000000...0.
        .0....1111111.0....

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testCubicNearNodeBudget() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

           A B C D E F G H J K L M N O P Q R S T
        19 . . . O O O O O O O O . . . . . O . O
        18 X X X . X X X X X . . . O O O O O . O
        17 O . . . . . O O . . . . . . . X X X .
        16 . . . . . . . O O O . . . . . . O . .
        15 O . O . . . . . . . . . . . . . . . .
        14 . . O . . . . . . O . . . . . . O . O
        13 . . O . . . . . . . X . . . . . O . .
        12 . . . . . . . . . O X . . . . . . . .
        11 . . . . . . . X . X . O . O . . . . .
        10 O X X . . . . . X . O . . . . . . . .
         9 O X . . . . . . X O . . . . . . . . .
         8 . . X O . . . . . X . . . . . . . . .
         7 O . . . X . . . . . . . . . . . . . .
         6 . . . . . . . . . . . . . . . . . . .
         5 . . . . . . . . . . . . . . . . . . .
         4 O . . . . . . . . . . . . . . . O . .
         3 . . . . . . . . . X O . . . . . O . .
         2 . . . O O . . . . X O X . X . . . X .
         1 . . . . . . X . O O O X . X . . O O O

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        ...00000000.....0.0
        000.00000...00000.0
        0.....00.......000.
        .......000......0..
        0.0................
        ..0......0......0.0
        ..0.......0.....0..
        .........00........
        .......0.0.0.0.....
        000.....0.0........
        00......01.........
        ..00.....0.........
        0...0..............
        ...................
        ...................
        0...............0..
        .........00.....0..
        ...00....000.0...0.
        ......0.0000.0..000

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testPolynomialMaxNodeBudget() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

           A B C D E F G H J K L M N O P Q R S T
        19 X . O O O O X . O O O O X . O O O O O
        18 O X X X X O O X X X X O O X X X X X O
        17 O X . . X X O X . . X X O X . . . X O
        16 O . . . . X O X . . . X O X . . . X O
        15 . . . . . . O X . . . . O X . . . X O
        14 . . . . . . . . . . . . . . . . . X O
        13 . . . . . . . . . . . . . . . X X O O
        12 . . . . . . . . . . . . . . . O O O X
        11 . . . . . . . . . . . . . . . . X X .
        10 . . . . . . . . . . . . . . . . . X O
         9 . . . . . . . . . . . . . . . . . X O
         8 . . . . . . . . . . . . . . . . . X O
         7 . . . . . . . . . . . . . . . X X O O
         6 . . . . . . . . . . . . . . . O O O X
         5 . . . . . . . . . . . . . . . . X X .
         4 . . . . . . . . . . . . . . . . . X O
         3 . . . . . . . . . . . . . . . . . X O
         2 . X . . . X . . . . . . . . . . X . O
         1 . . . . . . . . . . . . . . . O O O .

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printLadderMap(board1, defenderFirst: false)
        actual += "\n"
        let expected = #"""

        0.00000.00000.00000
        0000000000000000000
        00..0000..0000...00
        0....000...000...00
        ......00....00...00
        .................00
        ...............0000
        ...............0000
        ................00.
        .................00
        .................00
        .................00
        ...............0000
        ...............0000
        ................00.
        .................00
        .................00
        .0...0..........0.0
        ...............000.

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }
}
