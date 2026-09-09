import Darwin
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
  eval --model PATH --position FILE [--backend cpu|cpu-packed|metal|metal-packed|auto] [--dump-layers FILE]
                                 evaluate one position JSON (spatial/global/legal arrays) and print
                                 raw outputs and the post-processed evaluation as JSON; --dump-layers
                                 writes every logic layer's bits ([L,S,S,C] uint8) for parity checks
  gtp [--model-9 PATH] [--model-19 PATH] [--backend cpu|cpu-packed|metal|metal-packed|auto] [--visits N]
      [--value-source network|ownership|blend|rollout] [--value-blend W] [--value-k K] [--value-b B]
      [--rollout-count N] [--rollout-max-moves M]
                                 GTP engine on stdin/stdout (logs on stderr); at least one model
  selfplay --model PATH --games N --out DIR --seed N [--visits N] [--size 9|19]
           [--backend cpu|cpu-packed|metal|metal-packed|auto]
           [--value-source network|ownership|blend|rollout] [--value-blend W] [--value-k K] [--value-b B]
           [--rollout-count N] [--rollout-max-moves M]
                                 self-play with the model on both sides; writes SGF + root visit
                                 targets (JSONL) per game
  features --sgf-dir DIR --out FILE --size 9|19 [--rules cgos-area-psk-v1] [--komi K]
           [--max-games N] [--min-turn T]
                                 replay SGF main lines and stream one JSONL row per position;
                                 rejected games go to <out>.rejects.jsonl with file name and move
  benchmark --model PATH --positions FILE --batches 1,8,32 --out FILE.json
            [--backends cpu,cpu-packed,metal,metal-packed] [--warmup 50] [--iters 200]
                                 docs/spec/05-validation.md §7 Mac benchmark: warmup then measure
                                 each batch/backend combination on the first rows of the JSONL
                                 `ichigo features` produces (repeating rows if there are fewer than
                                 the largest batch). Writes FILE.json/FILE.csv (one row per
                                 batch/backend, exactly the §7 columns) and FILE.backend-profile.json
                                 (fastest measured backend per batch; docs/spec/04-tasks.md T25)

backends: cpu (byte gates + explicit-loop CPU heads, the golden oracle), cpu-packed (batch-packed
CPU gates + accelerated CPU heads), metal (byte gates on GPU + accelerated CPU heads),
metal-packed (batch-packed gates + heads, both on GPU). --backend auto picks metal if a Metal
device is available, else cpu. --backend metal/metal-packed with no device is a usage error
(exit 2), never a silent fallback.

value-source (docs/spec/03-engine.md §3-4): --value-source network (default) keeps the logic
network's own wdl head. ownership derives the search's leaf value from the ownership head instead
(value_own = sigmoid((score_est + b) / k), score_est = sum of ownership + signed komi); blend uses
weightNetwork*e_nn + (1-weightNetwork)*value_own. rollout runs --rollout-count (default 8)
policy-guided playouts per leaf (temperature 1 over the network's own legal-move policy, batched in
lock-step so each playout step is one evaluator call), scoring value_rollout = mean over playouts
of win=1/draw=0.5/loss=0 (white perspective) and blending weightNetwork*e_nn +
(1-weightNetwork)*value_rollout; --rollout-max-moves caps each playout (default 2*boardSize^2, e.g.
162 on 9x9), beyond which the position is scored as-is (an approximation) instead of via a natural
two-pass end — genmove logs the average playouts per leaf and the fraction that hit this cap.
--value-blend sets weightNetwork (default 0.5, used by both blend and rollout), --value-k/--value-b
set k/b for ownership/blend (defaults 6, 1.0). The raw network WDL is always kept and logged
alongside the search value regardless of this setting.
"""

/// cpu -> ScalarBackend, cpu-packed -> PackedCPUBackend, metal -> MetalBackend, metal-packed ->
/// MetalPackedBackend (both Metal backends: usage error, exit 2, if no Metal device), auto ->
/// metal if available else cpu (docs/spec/01-network.md §5: "Metal device がなければ...auto は
/// CPU"). The backend is picked once per process/session; there is no per-request runtime
/// fallback within one `gtp`/`selfplay` run.
func makeBackend(_ name: String, model: LogicModelData) -> any LogicBackend {
    switch name {
    case "cpu":
        return ScalarBackend(model: model)
    case "cpu-packed":
        return PackedCPUBackend(model: model)
    case "metal":
        do { return try MetalBackend(model: model) } catch { fail("metal backend unavailable: \(error)", .usage) }
    case "metal-packed":
        do { return try MetalPackedBackend(model: model) } catch { fail("metal-packed backend unavailable: \(error)", .usage) }
    case "auto":
        if model.manifest.gateArity == 4 {
            return PackedCPUBackend(model: model)
        }
        if MetalAvailability.probe().available {
            do { return try MetalBackend(model: model) } catch { fail("metal backend unavailable: \(error)", .usage) }
        }
        return ScalarBackend(model: model)
    default:
        fail("unknown --backend \(name) (expected cpu, cpu-packed, metal, metal-packed, or auto)", .usage)
    }
}

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
        "backends": metal.available ? ["cpu", "metal"] : ["cpu"],
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
    var hist: [Int: Int] = [:]
    for layer in model.gateTables { for g in layer { hist[Int(g), default: 0] += 1 } }
    printJSON([
        "boardSizes": m.boardSizes,
        "channels": m.channels,
        "layers": m.layers,
        "dilations": m.dilations,
        "calibrationTemperature": m.calibrationTemperature,
        "files": m.files.mapValues { ["byteLength": $0.byteLength, "sha256": $0.sha256] },
        "headTensors": m.headTensors.map { ["name": $0.name, "shape": $0.shape, "byteOffset": $0.byteOffset, "byteLength": $0.byteLength] },
        "payloadHash": model.payloadHash,
        "gateArity": m.gateArity,
        "gateEncoding": m.gateEncoding,
        "gateHistogram": Dictionary(uniqueKeysWithValues: hist.map { (String($0.key), $0.value) }),
        "trainingProvenance": (try? JSONSerialization.jsonObject(with: m.trainingProvenanceJSON.data(using: .utf8)!)) ?? [:],
    ])
}

/// Shared by `eval` (one position file) and `benchmark` (one row per line of an `ichigo
/// features` JSONL). Validates before converting: `UInt8(...)`/`Float(...)` trap on
/// out-of-range/non-finite input, so every value is checked first.
func parsePositionFields(_ obj: [String: Any], context: String) -> (size: Int, spatial: [UInt8], global: [Float], legal: [UInt8]) {
    guard let size = obj["boardSize"] as? Int,
          let spatialAny = obj["spatial"] as? [NSNumber], let globalAny = obj["global"] as? [NSNumber], let legalAny = obj["legal"] as? [NSNumber] else {
        fail("\(context): must be JSON with boardSize, spatial, global, legal", .usage)
    }
    func bits(_ values: [NSNumber], _ name: String) -> [UInt8] {
        values.map { v in
            let d = v.doubleValue
            guard d == 0 || d == 1 else { fail("\(context): \(name) values must be 0 or 1, got \(v)", .usage) }
            return UInt8(d)
        }
    }
    let spatial = bits(spatialAny, "spatial")
    let legal = bits(legalAny, "legal")
    let global = globalAny.map { v -> Float in
        let f = Float(v.doubleValue)
        guard v.doubleValue.isFinite, f.isFinite else { fail("\(context): global values must be finite numbers, got \(v)", .usage) }
        return f
    }
    return (size, spatial, global, legal)
}

/// `--dump-layers` works for any backend; `layerOutputs` isn't part of `LogicBackend` (Metal's
/// version is necessarily `async throws`, unlike the CPU scalar backend's), so dispatch on the
/// concrete type.
func dumpLayers(_ backend: any LogicBackend, features: FeatureBatch) async throws -> [[UInt8]] {
    if let scalar = backend as? ScalarBackend { return scalar.layerOutputs(features: features) }
    if let packed = backend as? PackedCPUBackend { return packed.layerOutputs(features: features) }
    if let metal = backend as? MetalBackend { return try await metal.layerOutputs(features: features) }
    if let metalPacked = backend as? MetalPackedBackend { return try await metalPacked.layerOutputs(features: features) }
    throw LogicModelError.backendUnavailable("--dump-layers is not supported for backend \(backend.name)")
}

func cmdEval(_ args: Args) async {
    let backendName = args.options["backend"] ?? "cpu"
    let model = loadModel(args.require("model"))
    let positionPath = args.require("position")
    guard let data = FileManager.default.contents(atPath: positionPath),
          let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        fail("position file must be valid JSON", .usage)
    }
    let (size, spatial, global, legal) = parsePositionFields(obj, context: "position file")
    let features: FeatureBatch
    do {
        features = try FeatureBatch(boardSize: size, batch: 1, spatial: spatial, global: global, legal: legal)
    } catch {
        fail("invalid position: \(error)", .usage)
    }
    let backend = makeBackend(backendName, model: model)
    if let dump = args.options["dump-layers"] {
        do {
            let layers = try await dumpLayers(backend, features: features)
            var dumpData = Data()
            for l in layers { dumpData.append(contentsOf: l) }
            guard FileManager.default.createFile(atPath: dump, contents: dumpData) else { fail("cannot write \(dump)", .usage) }
        } catch {
            fail("dump-layers failed: \(error)", .inference)
        }
    }
    do {
        let raw = try await backend.evaluate(features: features)
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

/// Shared by `gtp` and `selfplay`: `--value-source`/`--value-blend`/`--value-k`/`--value-b`/
/// `--rollout-count`/`--rollout-max-moves` (docs/spec/03-engine.md §3-4 "値ソース"). Parsing
/// itself lives in `ValueSourceFlags` (`IchiGoGTP`, unit-tested there) so this is just the
/// usage-error plumbing. `boardSize` only matters for `rollout`'s `--rollout-max-moves` default
/// (`2*boardSize^2`).
func parseValueSource(_ args: Args, boardSize: Int) -> ValueSource {
    do {
        return try ValueSourceFlags.parse(
            source: args.options["value-source"], blend: args.options["value-blend"], k: args.options["value-k"], b: args.options["value-b"],
            rolloutCount: args.options["rollout-count"], rolloutMaxMoves: args.options["rollout-max-moves"], boardSize: boardSize
        )
    } catch {
        fail("\(error)", .usage)
    }
}

func makeSlots(_ args: Args) -> [Int: GTPEngine.ModelSlot] {
    let backendName = args.options["backend"] ?? "auto"
    guard ["cpu", "cpu-packed", "metal", "metal-packed", "auto"].contains(backendName) else {
        fail("--backend \(backendName) must be cpu, cpu-packed, metal, metal-packed, or auto", .usage)
    }
    var slots: [Int: GTPEngine.ModelSlot] = [:]
    for (size, key) in [(9, "model-9"), (19, "model-19")] {
        guard let path = args.options[key] else { continue }
        let model = loadModel(path)
        guard model.manifest.boardSizes.contains(size) else { fail("\(path) does not support board size \(size)", .usage) }
        let ev = LogicEvaluator(model: model, backend: makeBackend(backendName, model: model))
        slots[size] = GTPEngine.ModelSlot(evaluator: ev, modelHash: model.payloadHash)
    }
    if let path = args.options["model"], slots.isEmpty {
        let model = loadModel(path)
        let backend = makeBackend(backendName, model: model) // one backend instance serves every board size the model declares
        for size in model.manifest.boardSizes {
            slots[size] = GTPEngine.ModelSlot(evaluator: LogicEvaluator(model: model, backend: backend), modelHash: model.payloadHash)
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
    cfg.searchSettings.valueSource = parseValueSource(args, boardSize: cfg.defaultBoardSize)
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
    var settings = SearchSettings()
    settings.valueSource = parseValueSource(args, boardSize: size)
    do {
            for g in 0 ..< games {
                let record = try await SelfPlay.playGame(slot: slot, size: size, komi: komi, visits: visits, seed: seed &+ UInt64(g), maxMoves: 4 * size * size, settings: settings)
                let base = (outDir as NSString).appendingPathComponent(String(format: "game-%05d", g))
                try record.sgf.write(toFile: base + ".sgf", atomically: true, encoding: .utf8)
                try record.targetsJSONL.write(toFile: base + ".targets.jsonl", atomically: true, encoding: .utf8)
                FileHandle.standardError.write("game \(g): \(record.moves) moves, result \(record.result), model \(slot.modelHash.prefix(12))\n".data(using: .utf8)!)
            }
    } catch { fail("selfplay failed: \(error)", .inference) }
}

// MARK: - benchmark (docs/spec/05-validation.md §7, docs/spec/04-tasks.md T25)

private extension Duration {
    var milliseconds: Double { Double(components.seconds) * 1000 + Double(components.attoseconds) * 1e-15 }
}

/// Streams the first `count` lines of a (potentially huge, e.g. 600MB+) JSONL file without
/// loading the whole thing into memory.
func readFirstLines(path: String, count: Int) -> [String] {
    guard let fh = FileHandle(forReadingAtPath: path) else { fail("cannot open \(path)", .usage) }
    defer { try? fh.close() }
    var buffer = Data()
    var lines: [String] = []
    let newline: UInt8 = 0x0A
    let chunkSize = 1 << 20
    while lines.count < count {
        let chunk = fh.readData(ofLength: chunkSize)
        if chunk.isEmpty { break }
        buffer.append(chunk)
        while lines.count < count, let idx = buffer.firstIndex(of: newline) {
            let lineData = buffer[buffer.startIndex ..< idx]
            if !lineData.isEmpty, let line = String(data: lineData, encoding: .utf8) { lines.append(line) }
            buffer.removeSubrange(buffer.startIndex ... idx)
        }
    }
    if lines.count < count, !buffer.isEmpty, let line = String(data: buffer, encoding: .utf8), !line.isEmpty {
        lines.append(line)
    }
    return lines
}

/// Concatenates `batch` rows (repeating from the top if `rows.count < batch`, per the task:
/// "take the first N rows for batch N, repeat rows if fewer") into one `FeatureBatch`.
func buildBatch(rows: [(size: Int, spatial: [UInt8], global: [Float], legal: [UInt8])], size: Int, batch: Int) throws -> FeatureBatch {
    var spatial: [UInt8] = []
    var global: [Float] = []
    var legal: [UInt8] = []
    spatial.reserveCapacity(batch * size * size * FeatureLayout.spatialChannels)
    global.reserveCapacity(batch * FeatureLayout.globalFeatures)
    legal.reserveCapacity(batch * (size * size + 1))
    for i in 0 ..< batch {
        let row = rows[i % rows.count]
        spatial.append(contentsOf: row.spatial)
        global.append(contentsOf: row.global)
        legal.append(contentsOf: row.legal)
    }
    return try FeatureBatch(boardSize: size, batch: batch, spatial: spatial, global: global, legal: legal)
}

func peakRSSBytes() -> UInt64 {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    // macOS ru_maxrss is already in bytes (unlike Linux, where it is kilobytes).
    return UInt64(usage.ru_maxrss)
}

func hardwareString() -> String {
    if let name = MetalAvailability.probe().deviceName { return name }
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    guard size > 0 else { return "unknown" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname("hw.model", &buf, &size, nil, 0) == 0 else { return "unknown" }
    let bytes = buf.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
    return String(decoding: bytes, as: UTF8.self)
}

func toolchainString() -> String {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    proc.arguments = ["swift", "--version"]
    let out = Pipe()
    proc.standardOutput = out
    proc.standardError = Pipe()
    do {
        try proc.run()
        proc.waitUntilExit()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        if let text = String(data: data, encoding: .utf8), let firstLine = text.split(separator: "\n", omittingEmptySubsequences: true).first {
            return firstLine.trimmingCharacters(in: .whitespaces)
        }
    } catch {
        // fall through to "unknown"
    }
    return "unknown"
}

func average(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count) }

/// Nearest-rank percentile (`p` in `[0,1]`) over a non-empty sample.
func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let rank = max(1, Int((p * Double(sorted.count)).rounded(.up)))
    return sorted[min(rank, sorted.count) - 1]
}

/// Runs the gate layers (gate_ms), the heads (head_ms) and `Postprocess` (post_ms) as three
/// separately-timed stages, bypassing `LogicBackend.evaluate`'s single-shot timing so the
/// benchmark can report each stage. `cpu` keeps the explicit-loop `Heads.evaluate` (the golden
/// oracle, docs/spec/01-network.md §5: "最適化前の golden oracle として永久に保持"); `cpu-packed`
/// and `metal` (byte gates + CPU heads) use the accelerated CPU head path
/// (`Heads.evaluateAccelerated`, docs/spec/04-tasks.md T24); `metal-packed` runs heads on the GPU
/// via `MetalPackedBackend.evaluateTimed`, which encodes gates+heads as two separate command
/// buffers purely so this benchmark can split their timing (the real `evaluate()` path fuses them
/// into one command buffer, per T24).
func runOneIteration(backend: any LogicBackend, model: LogicModelData, features: FeatureBatch) async throws -> (gateMs: Double, headMs: Double, postMs: Double) {
    let clock = ContinuousClock()

    if let metalPacked = backend as? MetalPackedBackend {
        let (raw, gateMs, headMs) = try await metalPacked.evaluateTimed(features: features)
        let t0 = clock.now
        if features.batch > 0 { _ = try Postprocess.evaluate(raw: raw, features: features) }
        let postMs = (clock.now - t0).milliseconds
        return (gateMs, headMs, postMs)
    }

    var t0 = clock.now
    let lastLayerBits: [UInt8]
    let useAcceleratedHeads: Bool
    if let scalar = backend as? ScalarBackend {
        let layers = scalar.layerOutputs(features: features)
        lastLayerBits = layers[model.layers - 1]
        useAcceleratedHeads = false
    } else if let packed = backend as? PackedCPUBackend {
        // Mirrors `PackedCPUBackend.evaluateSync`: only the last layer needs unpacking (unlike
        // `layerOutputs`, which unpacks every layer for parity tests).
        let packedLayers = packed.packedLayerOutputs(features: features)
        lastLayerBits = PackBits.unpack(packedLayers[model.layers - 1], boardSize: features.boardSize, batch: features.batch, channels: model.channels)
        useAcceleratedHeads = true
    } else if let metal = backend as? MetalBackend {
        let layers = try await metal.layerOutputs(features: features)
        lastLayerBits = layers[model.layers - 1]
        useAcceleratedHeads = true
    } else {
        throw LogicModelError.backendUnavailable("benchmark: unsupported backend \(backend.name)")
    }
    let gateMs = (clock.now - t0).milliseconds
    t0 = clock.now
    let raw = useAcceleratedHeads
        ? try Heads.evaluateAccelerated(model: model, lastLayer: lastLayerBits, features: features)
        : try Heads.evaluate(model: model, lastLayer: lastLayerBits, features: features)
    let headMs = (clock.now - t0).milliseconds
    t0 = clock.now
    if features.batch > 0 { _ = try Postprocess.evaluate(raw: raw, features: features) }
    let postMs = (clock.now - t0).milliseconds
    return (gateMs, headMs, postMs)
}

func cmdBenchmark(_ args: Args) async {
    let modelPath = args.require("model")
    let positionsPath = args.require("positions")
    let outPath = args.require("out")
    let batchesStr = args.require("batches")
    let batchTokens = batchesStr.split(separator: ",", omittingEmptySubsequences: true)
    let batches = batchTokens.compactMap { Int($0) }
    guard !batches.isEmpty, batches.count == batchTokens.count, batches.allSatisfy({ $0 >= 0 }) else {
        fail("--batches must be a comma-separated list of non-negative integers, got \(batchesStr)", .usage)
    }
    let knownBackends = ["cpu", "cpu-packed", "metal", "metal-packed"]
    let requestedBackends: [String]
    if let s = args.options["backends"] {
        let tokens = s.split(separator: ",", omittingEmptySubsequences: true).map(String.init)
        guard !tokens.isEmpty, tokens.allSatisfy({ knownBackends.contains($0) }) else {
            fail("--backends entries must be one of \(knownBackends.joined(separator: ",")), got \(s)", .usage)
        }
        requestedBackends = tokens
    } else {
        requestedBackends = MetalAvailability.probe().available ? knownBackends : ["cpu", "cpu-packed"]
    }
    let warmup = args.options["warmup"].flatMap(Int.init) ?? 50
    let iters = args.options["iters"].flatMap(Int.init) ?? 200
    guard warmup >= 0, iters >= 1 else { fail("--warmup must be >= 0 and --iters must be >= 1", .usage) }

    let clock = ContinuousClock()
    var t0 = clock.now
    let model = loadModel(modelPath)
    let modelLoadMs = (clock.now - t0).milliseconds
    guard let targetSize = model.manifest.boardSizes.first else { fail("model declares no board sizes", .usage) }

    let maxBatch = max(batches.max() ?? 0, 1)
    let lines = readFirstLines(path: positionsPath, count: maxBatch)
    guard !lines.isEmpty else { fail("--positions \(positionsPath) has no rows", .usage) }
    var rows: [(size: Int, spatial: [UInt8], global: [Float], legal: [UInt8])] = []
    for (i, line) in lines.enumerated() {
        guard let lineData = line.data(using: .utf8), let obj = (try? JSONSerialization.jsonObject(with: lineData)) as? [String: Any] else {
            fail("positions file line \(i): not valid JSON", .usage)
        }
        let fields = parsePositionFields(obj, context: "positions file line \(i)")
        guard fields.size == targetSize else {
            fail("positions file line \(i): boardSize \(fields.size) != model boardSize \(targetSize)", .usage)
        }
        rows.append(fields)
    }
    if rows.count < maxBatch {
        FileHandle.standardError.write("benchmark: --positions has only \(rows.count) rows, repeating to fill batches up to \(maxBatch)\n".data(using: .utf8)!)
    }

    var backends: [(name: String, backend: any LogicBackend, loadMs: Double)] = []
    for name in requestedBackends {
        t0 = clock.now
        let backend: any LogicBackend
        switch name {
        case "cpu":
            backend = ScalarBackend(model: model)
        case "cpu-packed":
            backend = PackedCPUBackend(model: model)
        case "metal":
            do { backend = try MetalBackend(model: model) } catch { fail("metal backend unavailable: \(error)", .usage) }
        case "metal-packed":
            do { backend = try MetalPackedBackend(model: model) } catch { fail("metal-packed backend unavailable: \(error)", .usage) }
        default:
            fail("unreachable --backends entry \(name)", .usage)
        }
        let backendInitMs = (clock.now - t0).milliseconds
        backends.append((name, backend, modelLoadMs + backendInitMs))
    }

    let hardware = hardwareString()
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    let toolchain = toolchainString()
    let columns = [
        "model_hash", "size", "batch", "backend", "samples", "feature_ms", "pack_ms", "gate_ms", "head_ms", "post_ms",
        "total_p50_ms", "total_p95_ms", "positions_per_sec", "peak_rss_bytes", "metal_allocated_bytes", "load_ms",
        "hardware", "os", "toolchain",
    ]
    var rowsOut: [[String: Any]] = []
    var measurementsForProfile: [(batch: Int, backend: String, positionsPerSec: Double)] = []

    do {
        for (backendName, backend, loadMs) in backends {
            for b in batches {
                let features = try buildBatch(rows: rows, size: targetSize, batch: b)
                for _ in 0 ..< warmup { _ = try await runOneIteration(backend: backend, model: model, features: features) }

                var gateTimes: [Double] = [], headTimes: [Double] = [], postTimes: [Double] = [], totalTimes: [Double] = []
                gateTimes.reserveCapacity(iters); headTimes.reserveCapacity(iters); postTimes.reserveCapacity(iters); totalTimes.reserveCapacity(iters)
                for _ in 0 ..< iters {
                    let (g, h, p) = try await runOneIteration(backend: backend, model: model, features: features)
                    gateTimes.append(g); headTimes.append(h); postTimes.append(p); totalTimes.append(g + h + p)
                }

                let samples = b * iters
                let totalElapsedMs = totalTimes.reduce(0, +)
                let positionsPerSec = totalElapsedMs > 0 ? Double(samples) / (totalElapsedMs / 1000) : 0
                let metalAllocated: UInt64 = (backend as? MetalBackend)?.currentAllocatedBytes ?? (backend as? MetalPackedBackend)?.currentAllocatedBytes ?? 0
                let row: [String: Any] = [
                    "model_hash": model.payloadHash, "size": targetSize, "batch": b, "backend": backendName, "samples": samples,
                    "feature_ms": 0.0, "pack_ms": 0.0,
                    "gate_ms": average(gateTimes), "head_ms": average(headTimes), "post_ms": average(postTimes),
                    "total_p50_ms": percentile(totalTimes, 0.50), "total_p95_ms": percentile(totalTimes, 0.95),
                    "positions_per_sec": positionsPerSec,
                    "peak_rss_bytes": Int(peakRSSBytes()), "metal_allocated_bytes": Int(metalAllocated),
                    "load_ms": loadMs, "hardware": hardware, "os": os, "toolchain": toolchain,
                ]
                rowsOut.append(row)
                measurementsForProfile.append((batch: b, backend: backendName, positionsPerSec: positionsPerSec))
                FileHandle.standardError.write(
                    "benchmark: backend=\(backendName) batch=\(b) samples=\(samples) total_p50_ms=\(row["total_p50_ms"]!) total_p95_ms=\(row["total_p95_ms"]!) positions_per_sec=\(positionsPerSec)\n"
                        .data(using: .utf8)!
                )
            }
        }
    } catch {
        fail("benchmark failed: \(error)", .inference)
    }

    let outURL = URL(fileURLWithPath: outPath)
    try? FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard JSONSerialization.isValidJSONObject(rowsOut),
          let jsonData = try? JSONSerialization.data(withJSONObject: rowsOut, options: [.prettyPrinted, .sortedKeys]) else {
        fail("could not serialise benchmark results", .inference)
    }
    guard FileManager.default.createFile(atPath: outPath, contents: jsonData) else { fail("cannot write \(outPath)", .usage) }

    func csvField(_ v: Any) -> String {
        if let s = v as? String { return "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\"" }
        if let d = v as? Double { return String(format: "%.6f", d) }
        return "\(v)"
    }
    var csv = columns.joined(separator: ",") + "\n"
    for row in rowsOut { csv += columns.map { csvField(row[$0]!) }.joined(separator: ",") + "\n" }
    let csvPath = outPath.hasSuffix(".json") ? String(outPath.dropLast(5)) + ".csv" : outPath + ".csv"
    try? csv.write(toFile: csvPath, atomically: true, encoding: .utf8)

    let winners = BackendSelector.fastestPerBatch(measurements: measurementsForProfile)
    let profile = BackendProfile(modelHash: model.payloadHash, hardware: hardware, batches: winners)
    if let profileData = try? JSONSerialization.data(withJSONObject: profile.toJSONObject(), options: [.prettyPrinted, .sortedKeys]) {
        let profilePath = outPath.hasSuffix(".json") ? String(outPath.dropLast(5)) + ".backend-profile.json" : outPath + ".backend-profile.json"
        try? profileData.write(to: URL(fileURLWithPath: profilePath))
    }
    FileHandle.standardError.write("benchmark: wrote \(outPath), \(csvPath), and the backend-profile file\n".data(using: .utf8)!)
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
case "eval": await cmdEval(args)
case "features": cmdFeatures(args)
case "gtp": await cmdGTP(args)
case "selfplay": await cmdSelfplay(args)
case "benchmark": await cmdBenchmark(args)
case "help", "--help", "-h": print(usage)
default:
    print(usage)
    fail("unknown command \(command)", .usage)
}
