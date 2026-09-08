import IchiGoCore
import XCTest

final class BoardAreaReferenceTests: XCTestCase {
    func testArea1() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        ..o.o.xx.
        .oooo.x.x
        oo.....xx
        .........
        .........
        x......oo
        .xx...o.o
        x.x..oo.o
        .x.x.o.o.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        ......XXX
        ......XXX
        .......XX
        .........
        .........
        .......OO
        ......OOO
        .....OOOO
        .....OOOO

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        ......XXX
        ......XXX
        .......XX
        .........
        .........
        .......OO
        ......OOO
        .....OOOO
        .....OOOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        ......XXX
        ......XXX
        .......XX
        .........
        .........
        .......OO
        ......OOO
        .....OOOO
        .....OOOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        ......XXX
        ......XXX
        .......XX
        .........
        .........
        .......OO
        ......OOO
        .....OOOO
        .....OOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OO.O..XXX
        O.....XXX
        .......XX
        .........
        .........
        .......OO
        X.....OOO
        .X...OOOO
        X.X..OOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        OO.O..XXX
        O.....XXX
        .......XX
        .........
        .........
        .......OO
        X.....OOO
        .X...OOOO
        X.X..OOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOOOO.XXX
        OOOOO.XXX
        OO.....XX
        .........
        .........
        X......OO
        XXX...OOO
        XXX..OOOO
        XXXX.OOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        OOOOO.XXX
        OOOOO.XXX
        OO.....XX
        .........
        .........
        X......OO
        XXX...OOO
        XXX..OOOO
        XXXX.OOOO

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testArea2() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        x.oooooo.
        oox..xx.o
        o...xox.o
        o...x.x.o
        oxxx.xx.o
        ox..x...o
        o.xox...o
        o.xxx...o
        .ooooooo.

        """#)
        let board2 = try Board.parseBoard(9, 9, #"""

        ..oooooo.
        oox..xx.o
        o...xox.o
        o...x.x.o
        oxxx.xx.o
        ox..x...o
        o.x.x...o
        o.xxx...o
        .ooooooo.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        actual += "-----\n"
        actual += BoardReferenceTestSupport.printAreas(board2)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOOOOOOO
        OO...XX.O
        O...XXX.O
        O...XXX.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOOOOOOO
        OO...XX.O
        O...XXX.O
        O...XXX.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OOOOOOOOO
        OO...XX.O
        O...XXX.O
        O...XXX.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        ........O
        .........
        .........
        .........
        ....X....
        .........
        .........
        .........
        O.......O

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOOOOOOOO
        OOX..XX.O
        O...XXX.O
        O...XXX.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        X.OOOOOOO
        OOX..XX.O
        O...XOX.O
        O...X.X.O
        OXXXXXX.O
        OX..X...O
        O.XOX...O
        O.XXX...O
        OOOOOOOOO

        -----
        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........
        .........

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OO......O
        .........
        .........
        .........
        ....X....
        ..XX.....
        ...X.....
        .........
        O.......O

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        OO......O
        .........
        .........
        .........
        ....X....
        ..XX.....
        ...X.....
        .........
        O.......O

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOOOOOOOO
        OOX..XX.O
        O...XOX.O
        O...X.X.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        OOOOOOOOO
        OOX..XX.O
        O...XOX.O
        O...X.X.O
        OXXXXXX.O
        OXXXX...O
        O.XXX...O
        O.XXX...O
        OOOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testArea3() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

        o.o....xx......o..o
        .oo.xxx.x......ooo.
        xo..x..xx........oo
        oo.`xxx..`.....`...
        ...................
        .......xx..........
        ....xxx.x......xxx.
        ....xo.xx......x.x.
        ....xxx........x.x.
        xxx`....xxxxx..`x.x
        ..x.....x...x...xx.
        o.xxx...x...x....xx
        o.x.x...x...xxx....
        ..xxx...x...x.x....
        xxx.....xxxxxxx....
        oo.`.....`.....`...
        .o.....oo.oo......o
        .oooo.o.o.o.o.oooo.
        o..o.o.oo.oo.o.o.o.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOO................
        OOO................
        OO.................
        OO.................
        ...................
        .......XX..........
        ....XXXXX......XXX.
        ....XXXXX......XXX.
        ....XXX........XXX.
        XXX.............XXX
        ..X.............XXX
        ..XXX............XX
        ..XXX..............
        ..XXX..............
        XXX................
        ...................
        ..........OO.......
        ..........OOO.OOOO.
        ..........OOOOOOOO.

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...............XXX.
        ...............XXX.
        ...............XXX.
        ................XXX
        ................XXX
        .................XX
        ...................
        ...................
        ...................
        ...................
        ..........OO.......
        ..........OOO.OOOO.
        ..........OOOOOOOO.

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOO................
        OOO................
        OO.................
        OO.................
        ...................
        .......XX..........
        ....XXXXX......XXX.
        ....XXXXX......XXX.
        ....XXX........XXX.
        XXX.............XXX
        ..X.............XXX
        ..XXX............XX
        ..XXX..............
        ..XXX..............
        XXX................
        ...................
        ..........OO.......
        ..........OOO.OOOO.
        ..........OOOOOOOO.

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        ...................
        ...................
        ...................
        ...................
        ...................
        ...................
        ...............XXX.
        ...............XXX.
        ...............XXX.
        ................XXX
        ................XXX
        .................XX
        ...................
        ...................
        ...................
        ...................
        ..........OO.......
        ..........OOO.OOOO.
        ..........OOOOOOOO.

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OOO.............OO.
        OOO....X..........O
        OO...XX............
        OO.................
        ...................
        .......XX..........
        ....XXXXX......XXX.
        ....XXXXX......XXX.
        ....XXX........XXX.
        XXX.............XXX
        ..X......XXX....XXX
        ..XXX....XXX.....XX
        ..XXX....XXX.......
        ..XXX....XXX.X.....
        XXX................
        ...................
        O.........OO.......
        O......O..OOO.OOOOO
        .OO.O.O...OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        .O..............OO.
        .......X..........O
        .....XX............
        ...................
        ...................
        ...................
        .......X.......XXX.
        ...............XXX.
        ...............XXX.
        ................XXX
        .........XXX....XXX
        .........XXX.....XX
        ...X.....XXX.......
        .........XXX.X.....
        ...................
        ...................
        O.........OO.......
        O......O..OOO.OOOOO
        .OO.O.O...OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOO....XX......OOOO
        OOO.XXXXX......OOOO
        OO..XXXXX........OO
        OO..XXX............
        ...................
        .......XX..........
        ....XXXXX......XXX.
        ....XXXXX......XXX.
        ....XXX........XXX.
        XXX.....XXXXX...XXX
        ..X.....XXXXX...XXX
        O.XXX...XXXXX....XX
        O.XXX...XXXXXXX....
        ..XXX...XXXXXXX....
        XXX.....XXXXXXX....
        OO.................
        OO.....OO.OO......O
        OOOOO.OOO.OOO.OOOOO
        OOOOOOOOO.OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        OOO....XX......OOOO
        .OO.XXXXX......OOOO
        XO..XXXXX........OO
        OO..XXX............
        ...................
        .......XX..........
        ....XXXXX......XXX.
        ....XO.XX......XXX.
        ....XXX........XXX.
        XXX.....XXXXX...XXX
        ..X.....XXXXX...XXX
        O.XXX...XXXXX....XX
        O.XXX...XXXXXXX....
        ..XXX...XXXXXXX....
        XXX.....XXXXXXX....
        OO.................
        OO.....OO.OO......O
        OOOOO.OOO.OOO.OOOOO
        OOOOOOOOO.OOOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testArea4() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

        .x.x.xxxx.xxxx.x.x.
        x.xxxx..x.x..xxxx.x
        xx.x..x.x.x.x..x.x.
        x.x`xx.xx`xx.xx`x.x
        .x.xx.xx...xx.xx.x.
        .x.x...........x.x.
        x.x.............x.x
        .x...............x.
        ...................
        xxx`....xxxxx..`...
        ..x.....x...x.xxxxx
        oxxxx...x..xx.x....
        o.x.x...x.o.xxx....
        ..xxx...x...x.x....
        xxx.....xxxxxxxxxxx
        ...`.....`....o`...
        ooooo....ooooo.oooo
        .o..oo...o...ooo...
        o.o.xo...o.o.o.o.oo

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXXXXXXXX..........
        XXXXXX..X..........
        XX.X....X..........
        X......XX..........
        ......XX...........
        ...................
        ...................
        ...................
        ...................
        XXX.....XXXXX......
        XXX.....XXXXX.XXXXX
        XXXXX...XXXXX.X....
        XXXXX...XXXXXXX....
        XXXXX...XXXXXXX....
        XXX.....XXXXXXXXXXX
        ...................
        ...................
        ...................
        ...................

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXXXXXXXX..........
        XXXXXX..X..........
        XX.X....X..........
        X......XX..........
        ......XX...........
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
        ...................
        ...................
        ...................

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXXXXXXXX..........
        XXXXXX..X..........
        XX.X....X..........
        X......XX..........
        ......XX...........
        ...................
        ...................
        ...................
        ...................
        XXX.....XXXXX......
        XXX.....XXXXX.XXXXX
        XXXXX...XXXXX.XXXXX
        XXXXX...XXXXXXXXXXX
        XXXXX...XXXXXXXXXXX
        XXX.....XXXXXXXXXXX
        ...................
        ...................
        ...................
        ...................

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXXXXXXXX..........
        XXXXXX..X..........
        XX.X....X..........
        X......XX..........
        ......XX...........
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
        ...................
        ...................
        ...................

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        XXXXXXXXX.....X.X.X
        XXXXXXXXX..XX....X.
        XXXXXX.XX..X.XX.X.X
        XX.X..XXX...X..X.X.
        X.X...XX........X.X
        X.X.............X.X
        .X...............X.
        ...................
        ...................
        XXX.....XXXXX......
        XXX.....XXXXX.XXXXX
        XXXXX...XXXXX.XXXXX
        XXXXX...XXXXXXXXXXX
        XXXXX...XXXXXXXXXXX
        XXX.....XXXXXXXXXXX
        ...................
        ..............O....
        O.........OOO...OOO
        .O........O.O.O.O..

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        XXXXXXXXX.....X.X.X
        XXXXXXXXX..XX....X.
        XXXXXX.XX..X.XX.X.X
        XX.X..XXX...X..X.X.
        X.X...XX........X.X
        X.X.............X.X
        .X...............X.
        ...................
        ...................
        ...................
        ...................
        ...............XXXX
        ...X...........XXXX
        .............X.XXXX
        ...................
        ...................
        ..............O....
        O.........OOO...OOO
        .O........O.O.O.O..

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXX.XX...XX.XXXXX
        XXXX...........XXXX
        XXX.............XXX
        .X...............X.
        ...................
        XXX.....XXXXX......
        XXX.....XXXXX.XXXXX
        XXXXX...XXXXX.XXXXX
        XXXXX...XXXXXXXXXXX
        XXXXX...XXXXXXXXXXX
        XXX.....XXXXXXXXXXX
        ..............O....
        OOOOO....OOOOOOOOOO
        OO..OO...OOOOOOOOOO
        OOO.XO...OOOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXXXXXX.XXXXXXXXX
        XXXXX.XX...XX.XXXXX
        XXXX...........XXXX
        XXX.............XXX
        .X...............X.
        ...................
        XXX.....XXXXX......
        ..X.....X...X.XXXXX
        OXXXX...X..XX.XXXXX
        O.XXX...X.O.XXXXXXX
        ..XXX...X...XXXXXXX
        XXX.....XXXXXXXXXXX
        ..............O....
        OOOOO....OOOOOOOOOO
        OO..OO...OOOOOOOOOO
        OOO.XO...OOOOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testArea5() throws {
        let board1 = try Board.parseBoard(19, 13, #"""

        ...................
        ...................
        ...............xx..
        ........xxxxxxxx.x.
        .....xxxx....x..ox.
        ..xxx...x..o..xxxx.
        ..x..xxx........x..
        ..x.ox..xxxxx...x..
        ..x.oxxx.oo.xxxxx..
        ..x.ooo.x..x...x...
        ..xxx...xxxxxx.x...
        ....xxxxx.....xx...
        ...................

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        ...................
        ...................
        ...............XX..
        ........XXXXXXXXXX.
        .....XXXX....XXXXX.
        ..XXXXXXX.....XXXX.
        ..XXXXXX........X..
        ..XXXXXXXXXXX...X..
        ..XXXXXXXXXXXXXXX..
        ..XXXXXXXXXXXXXX...
        ..XXXXXXXXXXXXXX...
        ....XXXXX.....XX...
        ...................

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
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
        ...................
        ...................

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXX....XXXXXX
        XXXXXXXXX.....XXXXX
        XXXXXXXX........XXX
        XXXXXXXXXXXXX...XXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
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
        ...................
        ...................

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXX....XXXXXX
        XXXXXXXXX.....XXXXX
        XXXXXXXX........XXX
        XXXXXXXXXXXXX...XXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXX..XX
        XXXXXXXX..........X
        XXXXX.............X
        XX...XXX..........X
        XX...............XX
        XX....XX.........XX
        XX...............XX
        XX..........XXX.XXX
        XX............X.XXX
        XXXX.....XXXXX..XXX
        XXXXXXXXXXXXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXX....XXXXXX
        XXXXXXXXX..O..XXXXX
        XXXXXXXX........XXX
        XXXXXXXXXXXXX...XXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXX.XX
        XXXXXXXXX....X..OXX
        XXXXXXXXX..O..XXXXX
        XXX..XXX........XXX
        XXX.OXXXXXXXX...XXX
        XXX.OXXX.OO.XXXXXXX
        XXX.OOO.X..XXXXXXXX
        XXXXX...XXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testAreaRect() throws {
        let board1 = try Board.parseBoard(12, 3, #"""

        x.ooxxxo.xx.
        oo.ox.xoox.x
        ooxox.xo.ox.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOO.....XXX
        OOOO.....XXX
        OOOO......XX

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........XXX
        .........XXX
        ..........XX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOO.....XXX
        OOOO.....XXX
        OOOO......XX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .........XXX
        .........XXX
        ..........XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OOOO.....XXX
        OOOO.X...XXX
        OOOO.X..O.XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        .........XXX
        .....X...XXX
        .....X..O.XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOOOXXXO.XXX
        OOOOXXXOOXXX
        OOOOXXXOOOXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        X.OOXXXO.XXX
        OO.OXXXOOXXX
        OOXOXXXOOOXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testPassAliveBugDerived() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

           A B C D E F G H J
         9 . O . O O X X O .
         8 O . O . O O X X O
         7 O O O O O O X . X
         6 O X O . O O X X X
         5 X X O O O O X X .
         4 . X O O O X X X .
         3 . X O O O O O O X
         2 O X X O X X X X X
         1 . X X X . X . . .

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        OOOOOXXXX
        OOOOOOXXX
        OOOOOOXXX
        OXOOOOXXX
        XXOOOOXXX
        XXOOOXXXX
        XXOOOOOOX
        XXXOXXXXX
        XXXXXXXXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testOneMoreSimple() throws {
        let board1 = try Board.parseBoard(7, 7, #"""

        ..ooo..
        ..xxx..
        xxxxxxx
        .....x.
        ooooox.
        .o..oxx
        .x..ox.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        .......
        ..XXX..
        XXXXXXX
        .....XX
        .....XX
        .....XX
        .....XX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        ..OOO..
        ..XXX..
        XXXXXXX
        .....XX
        OOOOOXX
        .O..OXX
        .X..OXX

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        ..OOO..
        ..XXX..
        XXXXXXX
        .....XX
        OOOOOXX
        .O..OXX
        .X..OXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testBugDerivedColorsOrientations() throws {
        let board1 = try Board.parseBoard(9, 9, #"""

        .oxo.oxo.
        ooxoooxxo
        xxxxxxxxx
        .x.x.....
        xxxoooooo
        ..o.xo.o.
        ooooooooo
        xooxxxoxx
        .xox.xox.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        XXXXXXXXX
        XXXXXXXXX
        XXXXXXXXX
        XXXX.....
        XXXOOOOOO
        ..OOOOOOO
        OOOOOOOOO
        OOOOOOOOO
        OOOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testRecursiveSafe() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

        .xx.....o.o.o...xx.
        x.x.x...ooooo.x.x.x
        xx.x..oo...o...x.xx
        ..x`.o.o.`.o...`x..
        .x.x..oo...o...x.x.
        .x.x....ooo....xox.
        o.x...oo...o....x..
        .x.x.o.o...o...x.x.
        ..x...oo...o....x..
        .xx`....o.o....`...
        xooo.....xxxx......
        .x....xxxx..xx.....
        x.....x.x....x.....
        ...o...x.x..xx.o.o.
        o.o.oo.xxxxxxoo.o.o
        .o.o...xooooo..o.o.
        oox.o..xoxo.o.o.xoo
        o.oo..xxoxoxox.oo.o
        .oo..x..x.x.x...oo.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printAreas(board1)
        let expected = #"""

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XX.........O.....XX
        ...........O.......
        ...........O.......
        ...................
        ...................
        ...................
        ...................
        ...................
        .........XXXX......
        ......XXXX..XX.....
        ......XXX....X.....
        .......XXX..XX.....
        .......XXXXXX......
        .O.....X.........O.
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO.............OOO

        Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XX.........O.....XX
        ...........O.......
        ...........O.......
        ...................
        ...................
        ...................
        ...................
        ...................
        .........XXXX......
        ......XXXX..XX.....
        ......XXX....X.....
        .......XXX..XX.....
        .......XXXXXX......
        .O.....X.........O.
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO.............OOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 0
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XX.........O.....XX
        ...........O.......
        ...........O.......
        ...................
        ...................
        ...................
        ...................
        ...................
        .........XXXX......
        ......XXXXXXXX.....
        ......XXXXXXXX.....
        .......XXXXXXX.....
        .......XXXXXX......
        .O.....X.........O.
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO.............OOO

        Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Suicide 1
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XX.........O.....XX
        ...........O.......
        ...........O.......
        ...................
        ...................
        ...................
        ...................
        ...................
        .........XXXX......
        ......XXXXXXXX.....
        ......XXXXXXXX.....
        .......XXXXXXX.....
        .......XXXXXX......
        .O.....X.........O.
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO.............OOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 0
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XXX.....OOOO....XXX
        ......O.OOOO.......
        ..X.....OOOO.......
        ..X................
        ...................
        ..X...O.........X..
        ...................
        ...................
        .........XXXX......
        X.....XXXXXXXX.....
        ......XXXXXXXX.....
        .......XXXXXXX.....
        ...O...XXXXXX..O.O.
        OO.....X.........OO
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO...XX.X.X....OOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Suicide 1
        XXX.....OOOOO...XXX
        XXX.....OOOOO...XXX
        XXX.....OOOO....XXX
        ......O.OOOO.......
        ..X.....OOOO.......
        ..X................
        ...................
        ..X...O.........X..
        ...................
        ...................
        .........XXXX......
        X.....XXXXXXXX.....
        ......XXXXXXXX.....
        .......XXXXXXX.....
        ...O...XXXXXX..O.O.
        OO.....X.........OO
        OO.....X.........OO
        OOOO..XX.......OOOO
        OOO...XX.X.X....OOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 0
        XXX.....OOOOO...XXX
        XXX.X...OOOOO.X.XXX
        XXXX..OOOOOO...XXXX
        ..X..OOOOOOO....X..
        .XXX..OOOOOO...X.X.
        .XXX....OOO....XOX.
        O.X...OO...O....X..
        .XXX.OOO...O...XXX.
        ..X...OO...O....X..
        .XX.....O.O........
        XOOO.....XXXX......
        XX....XXXXXXXX.....
        X.....XXXXXXXX.....
        ...O...XXXXXXX.O.O.
        O.OOOO.XXXXXXOOOOOO
        OO.O...XOOOOO..O.OO
        OOX.O..XOXO.O.O.XOO
        OOOO..XXOXOXOX.OOOO
        OOO..XXXXXXXX...OOO

        Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Suicide 1
        XXX.....OOOOO...XXX
        XXX.X...OOOOO.X.XXX
        XXXX..OOOOOO...XXXX
        ..X..OOOOOOO....X..
        .XXX..OOOOOO...X.X.
        .XXX....OOO....XOX.
        O.X...OO...O....X..
        .XXX.OOO...O...XXX.
        ..X...OO...O....X..
        .XX.....O.O........
        XOOO.....XXXX......
        XX....XXXXXXXX.....
        X.....XXXXXXXX.....
        ...O...XXXXXXX.O.O.
        O.OOOO.XXXXXXOOOOOO
        OO.O...XOOOOO..O.OO
        OOX.O..XOXO.O.O.XOO
        OOOO..XXOXOXOX.OOOO
        OOO..XXXXXXXX...OOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife1() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

        .oooxooo.ox..xo.o..
        o.xoxo.xoox..xoooxx
        oooxxxooxxx..xxxxoo
        xxx.xxxxx.x.....xo.
        ..xxx....xx.....xoo
        ................xxx
        xxxxxxxxxxxxxx..xoo
        .............x..xo.
        ..oo.........x.xxox
        oo.o.......ooxxooox
        xoooo...ooooxoox.x.
        .xo.o...o.ox.xoxxxx
        xoooo...ooooxoooooo
        oo.........oooooooo
        xxxxxxxxxxxxxxxxxxx
        oooooox....xooooooo
        xxxo.ox....xo.ooxxx
        ..xooox.xx.xooo.x..
        ..xo.ox....xo.oox..

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOO...............
        OOOO...............
        OOO................
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
        ...................
        OOOOOO.............
        XXXOOO.............
        XXXOOO.............
        XXXOOO.............

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOO...............
        OOOO...............
        OOO................
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
        ...................
        OOOOOO.............
        XXXOOO.............
        XXXOOO.............
        XXXOOO.............

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOO....O..XX..O...
        OOOO.......XX......
        OOO........XX......
        ...X.....X.XXXXX..O
        XX...XXXX..XXXXX...
        XXXXXXXXXXXXXXXX...
        ..............XX...
        ..............XX...
        ..............X....
        ..O................
        O.................X
        OO.O.....O..X......
        O..................
        ...................
        ...................
        OOOOOO.XXXX........
        XXXOOO.XXXX..O.....
        XXXOOO.X..X......XX
        XXXOOO.XXXX..O...XX

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOO....O..XX..O...
        OOOO.......XX......
        OOO........XX......
        ...X.....X.XXXXX..O
        XX...XXXX..XXXXX...
        XXXXXXXXXXXXXXXX...
        ..............XX...
        ..............XX...
        ..............X....
        ..O................
        O.................X
        OO.O.....O..X......
        O..................
        ...................
        ...................
        OOOOOO.XXXX........
        XXXOOO.XXXX..O.....
        XXXOOO.X..X......XX
        XXXOOO.XXXX..O...XX

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOXOOO.OX..XO.O..
        OOOOXO.XOOX..XOOOXX
        OOOXXXOOXXX..XXXXOO
        XXX.XXXXX.X.....XO.
        ..XXX....XX.....XOO
        ................XXX
        XXXXXXXXXXXXXX..XOO
        .............X..XO.
        ..OO.........X.XXOX
        OO.O.......OOXXOOOX
        .OOOO...OOOOXOOX.X.
        ..O.O...O.OX.XOXXXX
        .OOOO...OOOOXOOOOOO
        OO.........OOOOOOOO
        XXXXXXXXXXXXXXXXXXX
        OOOOOOX....XOOOOOOO
        XXXOOOX....XO.OOXXX
        XXXOOOX.XX.XOOO.X..
        XXXOOOX....XO.OOX..

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOXOOO.OX..XO.O..
        OOOOXO.XOOX..XOOOXX
        OOOXXXOOXXX..XXXXOO
        XXX.XXXXX.X.....XO.
        ..XXX....XX.....XOO
        ................XXX
        XXXXXXXXXXXXXX..XOO
        .............X..XO.
        ..OO.........X.XXOX
        OO.O.......OOXXOOOX
        .OOOO...OOOOXOOX.X.
        ..O.O...O.OX.XOXXXX
        .OOOO...OOOOXOOOOOO
        OO.........OOOOOOOO
        XXXXXXXXXXXXXXXXXXX
        OOOOOOX....XOOOOOOO
        XXXOOOX....XO.OOXXX
        XXXOOOX.XX.XOOO.X..
        XXXOOOX....XO.OOX..

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOXOOOOOXXXXOOO..
        OOOOXO.XOOXXXXOOOXX
        OOOXXXOOXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXOO
        .............XXXXO.
        ..OO.........XXXXOX
        OOOO.......OOXXOOOX
        OOOOO...OOOOXOOX.XX
        OOOOO...OOOXXXOXXXX
        OOOOO...OOOOXOOOOOO
        OO.........OOOOOOOO
        XXXXXXXXXXXXXXXXXXX
        OOOOOOXXXXXXOOOOOOO
        XXXOOOXXXXXXOOOOXXX
        XXXOOOXXXXXXOOO.XXX
        XXXOOOXXXXXXOOOOXXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOXOOOOOXXXXOOO..
        OOOOXO.XOOXXXXOOOXX
        OOOXXXOOXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXOO
        XXXXXXXXXXXXXXXXXXX
        XXXXXXXXXXXXXXXXXOO
        .............XXXXO.
        ..OO.........XXXXOX
        OOOO.......OOXXOOOX
        OOOOO...OOOOXOOX.XX
        OOOOO...OOOXXXOXXXX
        OOOOO...OOOOXOOOOOO
        OO.........OOOOOOOO
        XXXXXXXXXXXXXXXXXXX
        OOOOOOXXXXXXOOOOOOO
        XXXOOOXXXXXXOOOOXXX
        XXXOOOXXXXXXOOO.XXX
        XXXOOOXXXXXXOOOOXXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife2() throws {
        let board1 = try Board.parseBoard(19, 19, #"""

        x.o.ox.......xo.ox.
        xo..ox.......xo..ox
        o..ox.........xo..o
        ..ox.....o.....xo..
        oox.............xoo
        xx...............xx
        ...................
        ..xx.............xx
        xxoox......xxx..xoo
        oo.ox...xxxooox.xo.
        xoooox.xooooxooxxoo
        .xo.ox.xo.ox.xoxoo.
        xoooox.xooooxooxo.x
        ooxxx...xxxoooxxoo.
        xx.........xxx..xoo
        xxx..xxx.......xxxx
        oooxxooox...xxxoooo
        xxoooxxoox.xooo.oxx
        .xo.o.xo.oxxo.ooox.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 4
        ..............OOO..
        ..............OOOO.
        ...............OOOO
        ................OOO
        .................OO
        ...................
        ...................
        ...................
        ..OO.............OO
        OOOO.............OO
        OOOOO............OO
        OOOOO...........OOO
        OOOOO...........OOO
        OO..............OOO
        .................OO
        ...................
        ...............OOOO
        ............OOOOOOO
        ............OOOOOOO

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 3
        ..............OOO..
        ..............OOOO.
        ...............OOOO
        ................OOO
        .................OO
        ...................
        ...................
        ...................
        ..OO...............
        OOOO...............
        OOOOO..............
        OOOOO..............
        OOOOO..............
        OO.................
        ...................
        ...................
        ...............OOOO
        ............OOOOOOO
        ............OOOOOOO

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 4
        ...O..........OOO.X
        ..OO..........OOOO.
        .OO............OOOO
        OO..............OOO
        .................OO
        ...................
        ...................
        ...................
        ..OO.............OO
        OOOO.............OO
        OOOOO............OO
        OOOOO....O..X...OOO
        OOOOO...........OOO
        OO..............OOO
        .................OO
        ...................
        ...............OOOO
        ............OOOOOOO
        X..O....O...OOOOOOO

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 3
        ...O..........OOO.X
        ..OO..........OOOO.
        .OO............OOOO
        OO..............OOO
        .................OO
        ...................
        ...................
        ...................
        ..OO...............
        OOOO..............O
        OOOOO..............
        OOOOO....O..X......
        OOOOO..............
        OO.................
        ...................
        ...................
        ...............OOOO
        ............OOOOOOO
        X..O....O...OOOOOOO

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 4
        X.O.OX.......XOOOX.
        XO..OX.......XOOOOX
        O..OX.........XOOOO
        ..OX.....O.....XOOO
        OOX.............XOO
        XX...............XX
        ...................
        ..XX.............XX
        XXOOX......XXX..XOO
        OOOOX...XXXOOOX.XOO
        OOOOOX.XOOOOXOOXXOO
        OOOOOX.XO.OX.XOXOOO
        OOOOOX.XOOOOXOOXOOO
        OOXXX...XXXOOOXXOOO
        XX.........XXX..XOO
        XXX..XXX.......XXXX
        OOOXXOOOX...XXXOOOO
        XXOOOXXOOX.XOOOOOOO
        .XO.O.XO.OXXOOOOOOO

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 3
        X.O.OX.......XOOOX.
        XO..OX.......XOOOOX
        O..OX.........XOOOO
        ..OX.....O.....XOOO
        OOX.............XOO
        XX...............XX
        ...................
        ..XX.............XX
        XXOOX......XXX..XOO
        OOOOX...XXXOOOX.XO.
        OOOOOX.XOOOOXOOXXOO
        OOOOOX.XO.OX.XOXOO.
        OOOOOX.XOOOOXOOXO.X
        OOXXX...XXXOOOXXOO.
        XX.........XXX..XOO
        XXX..XXX.......XXXX
        OOOXXOOOX...XXXOOOO
        XXOOOXXOOX.XOOOOOOO
        .XO.O.XO.OXXOOOOOOO

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 4
        X.OOOX.......XOOOXX
        XOOOOX.......XOOOOX
        OOOOX.........XOOOO
        OOOX.....O.....XOOO
        OOX.............XOO
        XX...............XX
        ...................
        ..XX.............XX
        XXOOX......XXX..XOO
        OOOOX...XXXOOOX.XOO
        OOOOOX.XOOOOXOOXXOO
        OOOOOX.XOOOXXXOXOOO
        OOOOOX.XOOOOXOOXOOO
        OOXXX...XXXOOOXXOOO
        XX.........XXX..XOO
        XXX..XXX.......XXXX
        OOOXXOOOX...XXXOOOO
        XXOOOXXOOX.XOOOOOOO
        XXOOO.XOOOXXOOOOOOO

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 3
        X.OOOX.......XOOOXX
        XOOOOX.......XOOOOX
        OOOOX.........XOOOO
        OOOX.....O.....XOOO
        OOX.............XOO
        XX...............XX
        ...................
        ..XX.............XX
        XXOOX......XXX..XOO
        OOOOX...XXXOOOX.XOO
        OOOOOX.XOOOOXOOXXOO
        OOOOOX.XOOOXXXOXOO.
        OOOOOX.XOOOOXOOXO.X
        OOXXX...XXXOOOXXOO.
        XX.........XXX..XOO
        XXX..XXX.......XXXX
        OOOXXOOOX...XXXOOOO
        XXOOOXXOOX.XOOOOOOO
        XXOOO.XOOOXXOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife345() throws {
        let board1 = try Board.parseBoard(15, 4, #"""

        oooooox.x.xo...
        xxxxxoxxxxxoooo
        xoooxoooooxxxxx
        x.x.xo...oxx.x.

        """#)
        let board2 = try Board.parseBoard(15, 4, #"""

        oooooox.x.xo...
        xxxxxoxxxxxoooo
        xoooxooooooxxxx
        x.x.xo...oxx.x.

        """#)
        let board3 = try Board.parseBoard(15, 4, #"""

        oooooox.x.xo...
        xxxxxoxxxxooooo
        xoooxooooooxxxx
        x.x.xo...oxx.x.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board2)
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board3)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOXXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOXXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOXXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOXXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOXXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOXXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOXXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 1
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOXXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        .....OXXXXXOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXXOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOO.....OOOO
        .....O....OOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOO.....OOOO
        .....O....OOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOO.X.X.OOOO
        .....O....OOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOO.X.X.OOOO
        .....O....OOOOO
        .....OOOOOOXXXX
        .....OOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOX.X.XOOOO
        XXXXXOXXXXOOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOX.X.XOOOO
        XXXXXOXXXXOOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXOOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOOXXXXXOOOO
        XXXXXOXXXXOOOOO
        XOOOXOOOOOOXXXX
        X.X.XOOOOOXXXXX


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife6() throws {
        let board1 = try Board.parseBoard(15, 4, #"""

        .xo..ox.x.x.x..
        .xo..o.xoxx.xxx
        .xo...ooooxxooo
        .xo.......oo...

        """#)
        let board2 = try Board.parseBoard(15, 4, #"""

        .xo..ox.x.x.x..
        .xo..oxxoxx.xxx
        .xo...ooooxxooo
        .xo.......oo...

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board2)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        XX.............
        XX.............
        XX.............
        XX.............

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        XX.............
        XX.............
        XX.............
        XX.............

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        XX.OO..X.X.X.XX
        XX.OO......X...
        XX.OOO.........
        XX.OOOOOOO..OOO

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        XX.OO..X.X.X.XX
        XX.OO......X...
        XX.OOO.........
        XX.OOOOOOO..OOO

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        XXO..OX.X.X.X..
        XXO..O.XOXX.XXX
        XXO...OOOOXXOOO
        XXO.......OO...

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        XXO..OX.X.X.X..
        XXO..O.XOXX.XXX
        XXO...OOOOXXOOO
        XXO.......OO...

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        XXOOOOXXXXXXXXX
        XXOOOO.XOXXXXXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        XXOOOOXXXXXXXXX
        XXOOOO.XOXXXXXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOO.........
        XXOOOO..O......
        XXOOOOOOOO..OOO
        XXOOOOOOOOOOOOO

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOO.........
        XXOOOO..O......
        XXOOOOOOOO..OOO
        XXOOOOOOOOOOOOO

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOO.X.X.X.XX
        XXOOOO..O..X...
        XXOOOOOOOO..OOO
        XXOOOOOOOOOOOOO

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOO.X.X.X.XX
        XXOOOO..O..X...
        XXOOOOOOOO..OOO
        XXOOOOOOOOOOOOO

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOOX.X.X.X..
        XXOOOOXXOXX.XXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOOX.X.X.X..
        XXOOOOXXOXX.XXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOOXXXXXXXXX
        XXOOOOXXOXXXXXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        XXOOOOXXXXXXXXX
        XXOOOOXXOXXXXXX
        XXOOOOOOOOXXOOO
        XXOOOOOOOOOOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife7() throws {
        let board1 = try Board.parseBoard(15, 5, #"""

        .xo.oox....xo.o
        xxxooox....xoox
        .xo.oxx...xxxo.
        xoooox....xoooo
        ooxxxx....xo.o.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        ......XXXXXX...
        ......XXXXXX...
        .....XXXXXXXX..
        .....XXXXXX....
        ..XXXXXXXXX....

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        ......XXXXXX...
        ......XXXXXX...
        .....XXXXXXXX..
        .....XXXXXX....
        ..XXXXXXXXX....

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        X..O..XXXXXX.O.
        ......XXXXXX...
        X..O.XXXXXXXX..
        .....XXXXXX....
        ..XXXXXXXXX.O.O

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        X..O..XXXXXX.O.
        ......XXXXXX...
        X..O.XXXXXXXX..
        .....XXXXXX....
        ..XXXXXXXXX.O.O

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        .XO.OOXXXXXXO.O
        XXXOOOXXXXXXOOX
        .XO.OXXXXXXXXO.
        XOOOOXXXXXXOOOO
        OOXXXXXXXXXO.O.

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        .XO.OOXXXXXXO.O
        XXXOOOXXXXXXOOX
        .XO.OXXXXXXXXO.
        XOOOOXXXXXXOOOO
        OOXXXXXXXXXO.O.

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount -1
        XXOOOOXXXXXXOOO
        XXXOOOXXXXXXOOX
        XXOOOXXXXXXXXO.
        XOOOOXXXXXXOOOO
        OOXXXXXXXXXOOOO

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount -1
        XXOOOOXXXXXXOOO
        XXXOOOXXXXXXOOX
        XXOOOXXXXXXXXO.
        XOOOOXXXXXXOOOO
        OOXXXXXXXXXOOOO


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife8() throws {
        let board1 = try Board.parseBoard(15, 5, #"""

        ooo.ox.........
        .o.oox......xx.
        .ooox......x.ox
        oxxxx......x.oo
        oooox......x...

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ...O..XXXXXXXXX
        O.O...XXXXXX..X
        O....XXXXXX....
        .....XXXXXX....
        .....XXXXXX....

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ...O..XXXXXXXXX
        O.O...XXXXXX..X
        O....XXXXXX....
        .....XXXXXX....
        .....XXXXXX....

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOO.OX.........
        .O.OOX......XX.
        .OOOX......X.OX
        OXXXX......X.OO
        OOOOX......X...

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOO.OX.........
        .O.OOX......XX.
        .OOOX......X.OX
        OXXXX......X.OO
        OOOOX......X...

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOXXXXXXXXXX
        OOOOOXXXXXXXXXX
        OOOOXXXXXXXX.OX
        OXXXXXXXXXXX.OO
        OOOOXXXXXXXX...

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        OOOOOXXXXXXXXXX
        OOOOOXXXXXXXXXX
        OOOOXXXXXXXX.OX
        OXXXXXXXXXXX.OO
        OOOOXXXXXXXX...

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife9() throws {
        let board1 = try Board.parseBoard(15, 5, #"""

        x.x.ox.....oxx.
        oxx.oxx....oxox
        oooooox....oxoo
        ooxx.ox....ox..
        ..ox.ox....oxxx

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        .X............X
        ...............
        ...............
        ...............
        OO.............

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        .X............X
        ...............
        ...............
        ...............
        OO.............

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        X.X.OX.....OXX.
        OXX.OXX....OXOX
        OOOOOOX....OXOO
        OOXX.OX....OX..
        ..OX.OX....OXXX

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        X.X.OX.....OXX.
        OXX.OXX....OXOX
        OOOOOOX....OXOO
        OOXX.OX....OX..
        ..OX.OX....OXXX

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        XXX.OX.....OXXX
        OXX.OXX....OXOX
        OOOOOOX....OXOO
        OOXX.OX....OX..
        OOOX.OX....OXXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        XXX.OX.....OXXX
        OXX.OXX....OXOX
        OOOOOOX....OXOO
        OOXX.OX....OX..
        OOOX.OX....OXXX

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIndependentLife10() throws {
        let board1 = try Board.parseBoard(15, 5, #"""

        ............x.o
        ...........xxx.
        ........xxxx.xx
        ....oxxxxooxxxo
        ....x....xo.xo.

        """#)
        var actual = "\n"
        actual += BoardReferenceTestSupport.printIndependentLifeAreas(board1)
        let expected = #"""

        Keep Territories 0 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 0 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ...............
        ...............
        ...............

        Keep Territories 1 Keep Stones 0 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        .............XX
        ..............X
        ............X..
        ..............X
        .....XXXX....XX

        Keep Territories 1 Keep Stones 0 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ...............
        ...............
        ............X..
        ...............
        .....XXXX.....O

        Keep Territories 0 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ............X..
        ...........XXX.
        ........XXXX.XX
        ....OXXXXOOXXX.
        ....X....XO.X..

        Keep Territories 0 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ............X.O
        ...........XXX.
        ........XXXX.XX
        ....OXXXXOOXXXO
        ....X....XO.XO.

        Keep Territories 1 Keep Stones 1 Suicide 0
        whiteMinusBlackIndependentLifeRegionCount 0
        ............XXX
        ...........XXXX
        ........XXXXXXX
        ....OXXXXOOXXXX
        ....XXXXXXO.XXX

        Keep Territories 1 Keep Stones 1 Suicide 1
        whiteMinusBlackIndependentLifeRegionCount 0
        ............X.O
        ...........XXX.
        ........XXXXXXX
        ....OXXXXOOXXXO
        ....XXXXXXO.XOO

        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }

    func testIsNonPassAliveSelfConnection() throws {
        let board = try Board.parseBoard(9, 9, #"""

        .x.oo.xx.
        xoo.oxox.
        .o.ooooox
        ooo....x.
        ..o..x.xx
        oo.o...x.
        ...xxxoxx
        xxxx.x.x.
        o.o.xxoxo

        """#)
        let area = board.calculateArea(
            nonPassAliveStones: false,
            safeBigTerritories: true,
            unsafeBigTerritories: false,
            isMultiStoneSuicideLegal: false
        )
        var actual = "\n"
        for player in Player.allCases {
            actual += "NonPassAliveSelfConn \(player == .black ? "black" : "white")\n"
            for y in 0 ..< board.ySize {
                for x in 0 ..< board.xSize {
                    let loc = board.getLoc(x, y)
                    actual += board.colors[loc] == .empty
                        ? (board.isNonPassAliveSelfConnection(loc, player, area) ? "1" : "0")
                        : "-"
                }
                actual += "\n"
            }
            actual += "\n"
        }
        let expected = #"""


        NonPassAliveSelfConn black
        0-0--1--0
        ---0----1
        0-0------
        ---0000-1
        00-00-1--
        --0-010-0
        000------
        ----0-0-0
        -0-0-----

        NonPassAliveSelfConn white
        0-0--0--0
        ---0----0
        0-0------
        ---0000-0
        11-10-0--
        --1-000-0
        000------
        ----0-1-0
        -0-0-----


        """#
        BoardReferenceTestSupport.assertCppReference(actual, expected)
    }
}
