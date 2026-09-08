struct ZobristTables {
    let sizeX: [Hash128]
    let sizeY: [Hash128]
    let board: [[Hash128]]
    let board2: [[Hash128]]
    let player: [Hash128]
    let koLocation: [Hash128]
    let koMark: [[Hash128]]
    let encore: [Hash128]
    let secondEncoreStart: [[Hash128]]

    init() {
        var random = DeterministicRand(seed: "Board::initHash()")
        func nextHash(_ random: inout DeterministicRand) -> Hash128 {
            Hash128(random.nextUInt64(), random.nextUInt64())
        }
        var player = [Hash128](repeating: Hash128(), count: 4)
        for index in player.indices {
            player[index] = nextHash(&random)
        }
        var encore = [Hash128](repeating: Hash128(), count: 3)
        for index in encore.indices {
            encore[index] = nextHash(&random)
        }

        var board = [[Hash128]](repeating: [Hash128](repeating: Hash128(), count: 4), count: Board.maxArrSize)
        var koMark = board
        var koLocation = [Hash128](repeating: Hash128(), count: Board.maxArrSize)
        for loc in 0 ..< Board.maxArrSize {
            for color in [1, 2] {
                board[loc][color] = nextHash(&random)
                koMark[loc][color] = nextHash(&random)
            }
            koLocation[loc] = nextHash(&random)
        }

        random.initialize(seed: "Board::initHash() for ZOBRIST_SECOND_ENCORE_START hashes")
        var secondEncoreStart = [[Hash128]](
            repeating: [Hash128](repeating: Hash128(), count: 4),
            count: Board.maxArrSize
        )
        for loc in 0 ..< Board.maxArrSize {
            for color in [1, 2] {
                secondEncoreStart[loc][color] = nextHash(&random)
            }
        }

        random.initialize(seed: "Board::initHash() for ZOBRIST_SIZE hashes")
        var sizeX = [Hash128](repeating: Hash128(), count: Board.maxLen + 1)
        var sizeY = sizeX
        for size in 0 ... Board.maxLen {
            sizeX[size] = nextHash(&random)
            sizeY[size] = nextHash(&random)
        }

        random.initialize(seed: "Board::initHash() for second set of ZOBRIST hashes")
        var board2 = [[Hash128]](repeating: [Hash128](repeating: Hash128(), count: 4), count: Board.maxArrSize)
        for loc in 0 ..< Board.maxArrSize {
            for color in 0 ..< 4 {
                let hash = nextHash(&random)
                board2[loc][color] = Hash128(HashMix.murmurMix(hash.hash0), HashMix.splitMix64(hash.hash1))
            }
        }

        self.sizeX = sizeX
        self.sizeY = sizeY
        self.board = board
        self.board2 = board2
        self.player = player
        self.koLocation = koLocation
        self.koMark = koMark
        self.encore = encore
        self.secondEncoreStart = secondEncoreStart
    }
}
