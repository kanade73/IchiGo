import Foundation
import IchiGoCore
import XCTest

enum BoardReferenceTestSupport {
    static func printAreas(_ board: Board) -> String {
        let safeBigTerritories = [false, true, true, true]
        let unsafeBigTerritories = [false, false, true, true]
        let nonPassAliveStones = [false, false, false, true]
        var output = ""

        for mode in 0 ..< 8 {
            let suicide = mode % 2 == 1
            let index = mode / 2
            let area = board.calculateArea(
                nonPassAliveStones: nonPassAliveStones[index],
                safeBigTerritories: safeBigTerritories[index],
                unsafeBigTerritories: unsafeBigTerritories[index],
                isMultiStoneSuicideLegal: suicide
            )
            output += "Safe big territories \(cppBool(safeBigTerritories[index])) "
            output += "Unsafe big territories \(cppBool(unsafeBigTerritories[index])) "
            output += "Non pass alive stones \(cppBool(nonPassAliveStones[index])) "
            output += "Suicide \(cppBool(suicide))\n"
            output += printResultMap(board, area)
            output += "\n"
        }
        return output
    }

    static func printIndependentLifeAreas(_ board: Board) -> String {
        let keepTerritories = [false, true, false, true]
        let keepStones = [false, false, true, true]
        var output = ""

        for mode in 0 ..< 8 {
            let suicide = mode % 2 == 1
            let index = mode / 2
            let result = board.calculateIndependentLifeArea(
                keepTerritories: keepTerritories[index],
                keepStones: keepStones[index],
                isMultiStoneSuicideLegal: suicide
            )
            output += "Keep Territories \(cppBool(keepTerritories[index])) "
            output += "Keep Stones \(cppBool(keepStones[index])) "
            output += "Suicide \(cppBool(suicide))\n"
            output += "whiteMinusBlackIndependentLifeRegionCount "
            output += "\(result.whiteMinusBlackIndependentLifeRegionCount)\n"
            output += printResultMap(board, result.area)
            output += "\n"
        }
        return output
    }

    static func printLadderMap(_ board: Board, defenderFirst: Bool) -> String {
        var output = ""
        var buffer: [Loc] = []
        var workingMoves: [Loc] = []
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                let loc = board.getLoc(x, y)
                guard board.colors[loc] != .empty else {
                    output += "."
                    continue
                }
                let captured: Bool = if defenderFirst {
                    board.searchIsLadderCaptured(loc, defenderFirst: true, buffer: &buffer)
                } else {
                    board.searchIsLadderCapturedAttackerFirst2Libs(
                        loc,
                        buffer: &buffer,
                        workingMoves: &workingMoves
                    )
                }
                output += captured ? "1" : "0"
            }
            output += "\n"
        }
        return output
    }

    static func assertCppReference(
        _ actual: String,
        _ expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(normalizeCppReference(actual), normalizeCppReference(expected), file: file, line: line)
    }

    static func printResultMap(_ board: Board, _ result: [Color]) -> String {
        var output = ""
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                output.append(colorCharacter(result[board.getLoc(x, y)]))
            }
            output += "\n"
        }
        return output
    }

    private static func cppBool(_ value: Bool) -> Int {
        value ? 1 : 0
    }

    private static func colorCharacter(_ color: Color) -> Character {
        switch color {
        case .empty: "."
        case .black: "X"
        case .white: "O"
        case .wall: "#"
        }
    }

    private static func normalizeCppReference(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .joined(separator: "\n")
    }
}
