import Foundation
import IchiGoCore
import IchiGoEngine
import IchiGoFeatures
import IchiGoGTP
import LogicMetal
import LogicModel

// ichigo CLI (docs/spec/03-engine.md §6). M0 implements doctor, inspect, eval.
// Exit codes: 0 ok, 2 bad arguments / invalid model or input, 3 inference failure.

enum CLIExit: Int32 { case ok = 0, usage = 2, inference = 3 }

func fail(_ message: String, _ code: CLIExit) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code.rawValue)
}

func printJSON(_ object: Any) {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

struct Args {
    var positional: [String] = []
    var options: [String: String] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                    options[key] = argv[i + 1]
                    i += 2
                } else {
                    flags.insert(key)
                    i += 1
                }
            } else {
                positional.append(a)
                i += 1
            }
        }
    }

    func require(_ key: String) -> String {
        guard let v = options[key] else { fail("missing --\(key)", .usage) }
        return v
    }
}

let usage = """
usage: ichigo <command> [options]

commands:
  doctor                         CPU/OS/Metal capability report as JSON (no serial numbers)
  inspect --model PATH           validate a .ichigo directory, print manifest, shapes and hashes
  eval --model PATH --position FILE [--backend cpu] [--dump-layers FILE]
                                 evaluate one position JSON (spatial/global/legal arrays) and print
                                 raw outputs and the post-processed evaluation as JSON; --dump-layers
                                 writes every logic layer's bits ([L,S,S,C] uint8) for parity checks
  gtp [--model-9 PATH] [--model-19 PATH] [--backend cpu|auto] [--visits N]
                                 GTP engine on stdin/stdout (logs on stderr); at least one model
  selfplay --model PATH --games N --out DIR --seed N [--visits N] [--size 9|19]
                                 self-play with the model on both sides; writes SGF + root visit
                                 targets (JSONL) per game
  features --sgf-dir DIR --out FILE --size 9|19 [--rules cgos-area-psk-v1] [--komi K]
           [--max-games N] [--min-turn T]
                                 replay SGF main lines and stream one JSONL row per position;
                                 rejected games go to <out>.rejects.jsonl with file name and move
"""

func cmdFeatures(_ args: Args) {
    let dir = args.require("sgf-dir")
    let outPath = args.require("out")
    guard let size = Int(args.require("size")), size == 9 || size == 19 else { fail("--size must be 9 or 19", .usage) }
    let rules = args.options["rules"] ?? IchiGoRules.rulesID
    guard rules == IchiGoRules.rulesID else { fail("unsupported rules \(rules); only \(IchiGoRules.rulesID)", .usage) }
    let komiOverride = args.options["komi"].map { Float($0) ?? .nan }
    if let k = komiOverride, !k.isFinite { fail("--komi must be a number", .usage) }
    let maxGames = args.options["max-games"].flatMap(Int.init) ?? Int.max
    let minTurn = args.options["min-turn"].flatMap(Int.init) ?? 0
    let fm = FileManager.default
    guard let names = try? fm.contentsOfDirectory(atPath: dir) else { fail("cannot list \(dir)", .usage) }
    let sgfs = names.filter { $0.lowercased().hasSuffix(".sgf") }.sorted()
    guard fm.createFile(atPath: outPath, contents: nil), let out = FileHandle(forWritingAtPath: outPath) else { fail("cannot write \(outPath)", .usage) }
    let rejectPath = outPath + ".rejects.jsonl"
    _ = fm.createFile(atPath: rejectPath, contents: nil)
    let rejects = FileHandle(forWritingAtPath: rejectPath)!
    var games = 0, positions = 0, rejected = 0
    for name in sgfs.prefix(maxGames) {
        let path = (dir as NSString).appendingPathComponent(name)
        guard let data = fm.contents(atPath: path), let text = String(data: data, encoding: .utf8) else {
            rejected += 1
            rejects.write((try! JSONSerialization.data(withJSONObject: ["file": name, "reason": "not UTF-8"])) + "\n".data(using: .utf8)!)
            continue
        }
        do {
            let replay = try PositionExport.replay(sgf: text, expectedSize: size, komiOverride: komiOverride)
            for turn in minTurn ... replay.moves.count {
                var row = try PositionExport.row(replay, turn: turn)
                row["sourceFile"] = name
                out.write(try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes]))
                out.write("\n".data(using: .utf8)!)
                positions += 1
            }
            games += 1
        } catch {
            rejected += 1
            var reason: [String: Any] = ["file": name, "reason": "\(error)"]
            if case let PositionExport.RejectReason.illegalMove(index, _, _, _) = error { reason["moveIndex"] = index }
            rejects.write(try! JSONSerialization.data(withJSONObject: reason, options: [.sortedKeys]))
            rejects.write("\n".data(using: .utf8)!)
        }
    }
    try? out.close(); try? rejects.close()
    let summary: [String: Any] = ["games": games, "positions": positions, "rejected": rejected, "out": outPath, "rejects": rejectPath, "rulesId": rules, "boardSize": size, "featureVersion": FeatureEncoder.featureVersion]
    FileHandle.standardError.write((String(data: try! JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]), encoding: .utf8)! + "\n").data(using: .utf8)!)
}

func cmdDoctor() {
    let info = ProcessInfo.processInfo
    let metal = MetalAvailability.probe()
    var metalDict: [String: Any] = ["available": metal.available]
    if let n = metal.deviceName { metalDict["deviceName"] = n }
    if let m = metal.recommendedMaxWorkingSetBytes { metalDict["recommendedMaxWorkingSetBytes"] = m }
    if let u = metal.hasUnifiedMemory { metalDict["hasUnifiedMemory"] = u }
    var arch = "unknown"
    #if arch(arm64)
    arch = "arm64"
    #elseif arch(x86_64)
    arch = "x86_64"
    #endif
    printJSON([
        "os": "\(info.operatingSystemVersionString)",
        "architecture": arch,
        "cpuCores": info.activeProcessorCount,
        "physicalMemoryBytes": info.physicalMemory,
        "metal": metalDict,
        "featureVersion": FeatureEncoder.featureVersion,
        "backends": ["cpu"],
    ])
}

func loadModel(_ path: String) -> LogicModelData {
    do {
        return try ModelLoader.load(directory: URL(fileURLWithPath: path))
    } catch {
        fail("model rejected: \(error)", .usage)
    }
}

func cmdInspect(_ args: Args) {
    let model = loadModel(args.require("model"))
    let m = model.manifest
    var hist = [Int](repeating: 0, count: 16)
    for layer in model.gates { for g in layer { hist[Int(g)] += 1 } }
    printJSON([
        "boardSizes": m.boardSizes,
        "channels": m.channels,
        "layers": m.layers,
        "dilations": m.dilations,
        "calibrationTemperature": m.calibrationTemperature,
        "files": m.files.mapValues { ["byteLength": $0.byteLength, "sha256": $0.sha256] },
        "headTensors": m.headTensors.map { ["name": $0.name, "shape": $0.shape, "byteOffset": $0.byteOffset, "byteLength": $0.byteLength] },
        "payloadHash": model.payloadHash,
        "gateHistogram": hist,
        "trainingProvenance": (try? JSONSerialization.jsonObject(with: m.trainingProvenanceJSON.data(using: .utf8)!)) ?? [:],
    ])
}

func cmdEval(_ args: Args) {
    let backendName = args.options["backend"] ?? "cpu"
    guard backendName == "cpu" else { fail("backend \(backendName) is not implemented in M0 (only cpu)", .usage) }
    let model = loadModel(args.require("model"))
    let positionPath = args.require("position")
    guard let data = FileManager.default.contents(atPath: positionPath),
          let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let size = obj["boardSize"] as? Int,
          let spatialAny = obj["spatial"] as? [NSNumber], let globalAny = obj["global"] as? [NSNumber], let legalAny = obj["legal"] as? [NSNumber] else {
        fail("position file must be JSON with boardSize, spatial, global, legal", .usage)
    }
    // Validate before converting: UInt8(...) and Float(...) trap on out-of-range/non-finite input.
    func bits(_ values: [NSNumber], _ name: String) -> [UInt8] {
        values.map { v in
            let d = v.doubleValue
            guard d == 0 || d == 1 else { fail("invalid position: \(name) values must be 0 or 1, got \(v)", .usage) }
            return UInt8(d)
        }
    }
    let spatial = bits(spatialAny, "spatial")
    let legal = bits(legalAny, "legal")
    let global = globalAny.map { v -> Float in
        let f = Float(v.doubleValue)
        guard v.doubleValue.isFinite, f.isFinite else { fail("invalid position: global values must be finite numbers, got \(v)", .usage) }
        return f
    }
    let features: FeatureBatch
    do {
        features = try FeatureBatch(boardSize: size, batch: 1, spatial: spatial, global: global, legal: legal)
    } catch {
        fail("invalid position: \(error)", .usage)
    }
    let backend = ScalarBackend(model: model)
    if let dump = args.options["dump-layers"] {
        let layers = backend.layerOutputs(features: features)
        var data = Data()
        for l in layers { data.append(contentsOf: l) }
        guard FileManager.default.createFile(atPath: dump, contents: data) else { fail("cannot write \(dump)", .usage) }
    }
    do {
        let raw = try backend.evaluateSync(features: features)
        let post = try Postprocess.evaluate(raw: raw, features: features)[0]
        printJSON([
            "modelHash": model.payloadHash,
            "backend": backend.name,
            "boardSize": size,
            "raw": [
                "policyLogits": raw.policyLogits, "wdlLogits": raw.wdlLogits, "scoreMean": raw.scoreMean[0], "ownership": raw.ownership,
            ],
            "evaluation": [
                "policy": post.policy, "winDrawLoss": post.winDrawLoss, "expectedResult": post.expectedResult,
                "scoreMean": post.scoreMean, "ownership": post.ownership, "perspective": "to-move",
            ],
        ])
    } catch {
        fail("inference failed: \(error)", .inference)
    }
}

func makeSlots(_ args: Args) -> [Int: GTPEngine.ModelSlot] {
    let backendName = args.options["backend"] ?? "auto"
    guard backendName == "cpu" || backendName == "auto" else { fail("backend \(backendName) is not implemented (cpu only in this milestone)", .usage) }
    var slots: [Int: GTPEngine.ModelSlot] = [:]
    for (size, key) in [(9, "model-9"), (19, "model-19")] {
        guard let path = args.options[key] else { continue }
        let model = loadModel(path)
        guard model.manifest.boardSizes.contains(size) else { fail("\(path) does not support board size \(size)", .usage) }
        let ev = LogicEvaluator(model: model, backend: ScalarBackend(model: model))
        slots[size] = GTPEngine.ModelSlot(evaluator: ev, modelHash: model.payloadHash)
    }
    if let path = args.options["model"], slots.isEmpty {
        let model = loadModel(path)
        for size in model.manifest.boardSizes {
            slots[size] = GTPEngine.ModelSlot(evaluator: LogicEvaluator(model: model, backend: ScalarBackend(model: model)), modelHash: model.payloadHash)
        }
    }
    guard !slots.isEmpty else { fail("no model given (--model-9 / --model-19 / --model)", .usage) }
    return slots
}

func cmdGTP(_ args: Args) async {
    let slots = makeSlots(args)
    var cfg = GTPEngine.Config()
    cfg.visits = args.options["visits"].flatMap(Int.init) ?? 100
    cfg.defaultBoardSize = slots[9] != nil ? 9 : 19
    let engine: GTPEngine
    do {
        engine = try GTPEngine(models: slots, config: cfg, log: { msg in FileHandle.standardError.write(("[ichigo] " + msg + "\n").data(using: .utf8)!) })
    } catch { fail("cannot start engine: \(error)", .usage) }
    await GTPLoop.run(engine: engine)
}

func cmdSelfplay(_ args: Args) async {
    let slots = makeSlots(args)
    guard let games = args.options["games"].flatMap(Int.init), games > 0 else { fail("--games N required", .usage) }
    guard let seed = args.options["seed"].flatMap(UInt64.init) else { fail("--seed N required", .usage) }
    let outDir = args.require("out")
    let visits = args.options["visits"].flatMap(Int.init) ?? 100
    let size = args.options["size"].flatMap(Int.init) ?? (slots[9] != nil ? 9 : 19)
    guard let slot = slots[size] else { fail("no model for size \(size)", .usage) }
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let komi = size == 9 ? IchiGoRules.defaultKomi9 : IchiGoRules.defaultKomi19
    do {
            for g in 0 ..< games {
                let record = try await SelfPlay.playGame(slot: slot, size: size, komi: komi, visits: visits, seed: seed &+ UInt64(g), maxMoves: 4 * size * size)
                let base = (outDir as NSString).appendingPathComponent(String(format: "game-%05d", g))
                try record.sgf.write(toFile: base + ".sgf", atomically: true, encoding: .utf8)
                try record.targetsJSONL.write(toFile: base + ".targets.jsonl", atomically: true, encoding: .utf8)
                FileHandle.standardError.write("game \(g): \(record.moves) moves, result \(record.result), model \(slot.modelHash.prefix(12))\n".data(using: .utf8)!)
            }
    } catch { fail("selfplay failed: \(error)", .inference) }
}

let argv = Array(CommandLine.arguments.dropFirst())
guard let command = argv.first else {
    print(usage)
    exit(CLIExit.usage.rawValue)
}
let args = Args(Array(argv.dropFirst()))
switch command {
case "doctor": cmdDoctor()
case "inspect": cmdInspect(args)
case "eval": cmdEval(args)
case "features": cmdFeatures(args)
case "gtp": await cmdGTP(args)
case "selfplay": await cmdSelfplay(args)
case "help", "--help", "-h": print(usage)
default:
    print(usage)
    fail("unknown command \(command)", .usage)
}
