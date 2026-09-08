import Foundation

public enum RulesError: Error, Equatable {
    case invalidValue(String)
    case couldNotParse(String)
    case unknownOption(String)
}

public struct Rules: Equatable, Sendable, CustomStringConvertible {
    public enum KoRule: Int, CaseIterable, Sendable { case simple, positional, situational, spight }
    public enum ScoringRule: Int, CaseIterable, Sendable { case area, territory }
    public enum TaxRule: Int, CaseIterable, Sendable { case none, seki, all }
    public enum WhiteHandicapBonusRule: Int, CaseIterable, Sendable { case zero, n, nMinusOne }

    public static let minUserKomi: Float = -150
    public static let maxUserKomi: Float = 150

    public var koRule: KoRule
    public var scoringRule: ScoringRule
    public var taxRule: TaxRule
    public var multiStoneSuicideLegal: Bool
    public var hasButton: Bool
    public var whiteHandicapBonusRule: WhiteHandicapBonusRule
    public var friendlyPassOk: Bool
    public var komi: Float

    public init(
        koRule: KoRule = .positional,
        scoringRule: ScoringRule = .area,
        taxRule: TaxRule = .none,
        multiStoneSuicideLegal: Bool = true,
        hasButton: Bool = false,
        whiteHandicapBonusRule: WhiteHandicapBonusRule = .zero,
        friendlyPassOk: Bool = false,
        komi: Float = 7.5
    ) {
        self.koRule = koRule
        self.scoringRule = scoringRule
        self.taxRule = taxRule
        self.multiStoneSuicideLegal = multiStoneSuicideLegal
        self.hasButton = hasButton
        self.whiteHandicapBonusRule = whiteHandicapBonusRule
        self.friendlyPassOk = friendlyPassOk
        self.komi = komi
    }

    public static func getTrompTaylorish() -> Rules {
        Rules()
    }

    public static func getSimpleTerritory() -> Rules {
        Rules(
            koRule: .simple,
            scoringRule: .territory,
            taxRule: .seki,
            multiStoneSuicideLegal: false
        )
    }

    public func equalsIgnoringKomi(_ other: Rules) -> Bool {
        koRule == other.koRule && scoringRule == other.scoringRule && taxRule == other.taxRule
            && multiStoneSuicideLegal == other.multiStoneSuicideLegal && hasButton == other.hasButton
            && whiteHandicapBonusRule == other.whiteHandicapBonusRule && friendlyPassOk == other.friendlyPassOk
    }

    public func gameResultWillBeInteger() -> Bool {
        (komi.rounded(.towardZero) == komi) != hasButton
    }

    public static func komiIsIntOrHalfInt(_ komi: Float) -> Bool {
        komi.isFinite && komi * 2 == (komi * 2).rounded(.towardZero)
    }

    public static func koRuleStrings() -> Set<String> {
        Set(KoRule.allCases.map(writeKoRule))
    }

    public static func scoringRuleStrings() -> Set<String> {
        Set(ScoringRule.allCases.map(writeScoringRule))
    }

    public static func taxRuleStrings() -> Set<String> {
        Set(TaxRule.allCases.map(writeTaxRule))
    }

    public static func whiteHandicapBonusRuleStrings() -> Set<String> {
        Set(WhiteHandicapBonusRule.allCases.map(writeWhiteHandicapBonusRule))
    }

    public static func parseKoRule(_ string: String) throws -> KoRule {
        switch string {
        case "SIMPLE": .simple
        case "POSITIONAL": .positional
        case "SITUATIONAL": .situational
        case "SPIGHT": .spight
        default: throw RulesError.invalidValue("Invalid ko rule: \(string)")
        }
    }

    public static func parseScoringRule(_ string: String) throws -> ScoringRule {
        switch string {
        case "AREA": .area
        case "TERRITORY": .territory
        default: throw RulesError.invalidValue("Invalid scoring rule: \(string)")
        }
    }

    public static func parseTaxRule(_ string: String) throws -> TaxRule {
        switch string {
        case "NONE": .none
        case "SEKI": .seki
        case "ALL": .all
        default: throw RulesError.invalidValue("Invalid tax rule: \(string)")
        }
    }

    public static func parseWhiteHandicapBonusRule(_ string: String) throws -> WhiteHandicapBonusRule {
        switch string {
        case "0": .zero
        case "N": .n
        case "N-1": .nMinusOne
        default: throw RulesError.invalidValue("Invalid whiteHandicapBonus rule: \(string)")
        }
    }

    public static func writeKoRule(_ rule: KoRule) -> String {
        switch rule {
        case .simple: "SIMPLE"
        case .positional: "POSITIONAL"
        case .situational: "SITUATIONAL"
        case .spight: "SPIGHT"
        }
    }

    public static func writeScoringRule(_ rule: ScoringRule) -> String {
        switch rule {
        case .area: "AREA"
        case .territory: "TERRITORY"
        }
    }

    public static func writeTaxRule(_ rule: TaxRule) -> String {
        switch rule {
        case .none: "NONE"
        case .seki: "SEKI"
        case .all: "ALL"
        }
    }

    public static func writeWhiteHandicapBonusRule(_ rule: WhiteHandicapBonusRule) -> String {
        switch rule {
        case .zero: "0"
        case .n: "N"
        case .nMinusOne: "N-1"
        }
    }

    public var description: String {
        toString()
    }

    public func toString() -> String {
        toStringNoKomi() + "komi" + Self.formatFloat(komi)
    }

    public func toStringNoKomi() -> String {
        var result = "ko\(Self.writeKoRule(koRule))score\(Self.writeScoringRule(scoringRule))"
        result += "tax\(Self.writeTaxRule(taxRule))sui\(multiStoneSuicideLegal ? 1 : 0)"
        if hasButton { result += "button1" }
        if whiteHandicapBonusRule != .zero {
            result += "whb\(Self.writeWhiteHandicapBonusRule(whiteHandicapBonusRule))"
        }
        if friendlyPassOk { result += "fpok1" }
        return result
    }

    public func toStringNoKomiMaybeNice() -> String {
        let names = ["TrompTaylor", "Japanese", "Chinese", "Chinese-OGS", "AGA", "StoneScoring", "NewZealand"]
        for name in names {
            if let candidate = try? Self.parseRulesWithoutKomi(name, komi: komi), equalsIgnoringKomi(candidate) {
                return name
            }
        }
        return toStringNoKomi()
    }

    public func toJsonString() -> String {
        toJsonStringHelper(omitKomi: false, omitDefaults: false)
    }

    public func toJsonStringNoKomi() -> String {
        toJsonStringHelper(omitKomi: true, omitDefaults: false)
    }

    public func toJsonStringNoKomiMaybeOmitStuff() -> String {
        toJsonStringHelper(omitKomi: true, omitDefaults: true)
    }

    public static func parseRules(_ string: String) throws -> Rules {
        try parseRulesHelper(string, allowKomi: true)
    }

    public static func parseRulesWithoutKomi(_ string: String, komi: Float) throws -> Rules {
        var rules = try parseRulesHelper(string, allowKomi: false)
        rules.komi = komi
        return rules
    }

    public static func tryParseRules(_ string: String) -> Rules? {
        try? parseRules(string)
    }

    public static func tryParseRulesWithoutKomi(_ string: String, komi: Float) -> Rules? {
        try? parseRulesWithoutKomi(string, komi: komi)
    }

    public static func updateRules(
        key originalKey: String,
        value originalValue: String,
        priorRules: Rules
    ) throws -> Rules {
        var rules = priorRules
        let key = originalKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = originalValue.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        switch key {
        case "ko": rules.koRule = try parseKoRule(value)
        case "score", "scoring": rules.scoringRule = try parseScoringRule(value)
        case "tax": rules.taxRule = try parseTaxRule(value)
        case "suicide": rules.multiStoneSuicideLegal = try parseBool(value)
        case "hasButton": rules.hasButton = try parseBool(value)
        case "whiteHandicapBonus": rules.whiteHandicapBonusRule = try parseWhiteHandicapBonusRule(value)
        case "friendlyPassOk": rules.friendlyPassOk = try parseBool(value)
        default: throw RulesError.unknownOption(key)
        }
        return rules
    }

    public static let zobristKoRuleHash = [
        Hash128(0x3CC7_E0BF_8468_20F6, 0x1FB7_FBDE_5FC6_BA4E),
        Hash128(0xCC18_F5D4_7188_554A, 0x3A63_152C_23E4_128D),
        Hash128(0x3BC5_5E42_B23B_35BF, 0xC75F_A1E6_1562_1DCD),
        Hash128(0x5B20_96E4_8241_D21B, 0x23CC_18D4_E85C_D67F),
    ]
    public static let zobristScoringRuleHash = [
        Hash128(0x8B3E_D759_8F90_1494 ^ 0x72EE_CCC7_2C82_A5E7, 0x1DFD_47AC_77BC_E5F8 ^ 0x0D12_65E4_1362_3E2B),
        Hash128(0x3813_45DC_357E_C982 ^ 0x125B_FE48_A410_42D5, 0x03BA_55C0_2602_6B56 ^ 0x0618_66B5_F2B9_8A79),
    ]
    public static let zobristTaxRuleHash = [
        Hash128(0x72EE_CCC7_2C82_A5E7, 0x0D12_65E4_1362_3E2B),
        Hash128(0x125B_FE48_A410_42D5, 0x0618_66B5_F2B9_8A79),
        Hash128(0xA384_ECE9_D8EE_713C, 0xFDC9_F3B5_D1F3_732B),
    ]
    public static let zobristMultiStoneSuicideHash = Hash128(0xF9B4_75B3_BBF3_5E37, 0xEFA1_9D8B_1E5B_3E5A)
    public static let zobristButtonHash = Hash128(0xB8B9_14C9_234E_CE84, 0x3D75_9CDD_EBE2_9C14)
    public static let zobristFriendlyPassOkHash = Hash128(0x0113_6559_98EF_0A25, 0x99C9_D04E_CD96_4874)

    private static func parseRulesHelper(_ original: String, allowKomi: Bool) throws -> Rules {
        let lower = original.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let named = namedRules(lower) { return named }
        if original.first == "{" { return try parseJsonRules(original, allowKomi: allowKomi) }
        return try parseLegacyRules(original, allowKomi: allowKomi)
    }

    private static func namedRules(_ name: String) -> Rules? {
        switch name {
        case "japanese", "korean":
            Rules(koRule: .simple, scoringRule: .territory, taxRule: .seki, multiStoneSuicideLegal: false, komi: 6.5)
        case "chinese":
            Rules(koRule: .simple, multiStoneSuicideLegal: false, whiteHandicapBonusRule: .n, friendlyPassOk: true)
        case "chineseogs", "chinese_ogs", "chinese-ogs", "chinesekgs", "chinese_kgs", "chinese-kgs":
            Rules(multiStoneSuicideLegal: false, whiteHandicapBonusRule: .n, friendlyPassOk: true)
        case "ancientarea", "ancient-area", "ancient_area", "ancient area", "stonescoring", "stone-scoring",
             "stone_scoring",
             "stone scoring":
            Rules(koRule: .simple, taxRule: .all, multiStoneSuicideLegal: false, friendlyPassOk: true)
        case "ancientterritory", "ancient-territory", "ancient_territory", "ancient territory":
            Rules(koRule: .simple, scoringRule: .territory, taxRule: .all, multiStoneSuicideLegal: false, komi: 6.5)
        case "agabutton", "aga-button", "aga_button", "aga button":
            Rules(
                koRule: .situational,
                multiStoneSuicideLegal: false,
                hasButton: true,
                whiteHandicapBonusRule: .nMinusOne,
                friendlyPassOk: true,
                komi: 7
            )
        case "aga", "bga", "french":
            Rules(
                koRule: .situational,
                multiStoneSuicideLegal: false,
                whiteHandicapBonusRule: .nMinusOne,
                friendlyPassOk: true
            )
        case "nz", "newzealand", "new zealand", "new-zealand", "new_zealand":
            Rules(koRule: .situational, friendlyPassOk: true)
        case "tromp-taylor", "tromp_taylor", "tromp taylor", "tromptaylor": Rules()
        case "goe", "ing": Rules(friendlyPassOk: true)
        default: nil
        }
    }

    private static func parseJsonRules(_ original: String, allowKomi: Bool) throws -> Rules {
        guard let data = original.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any]
        else { throw RulesError.couldNotParse(original) }
        var rules = Rules()
        var komiSpecified = false
        var taxSpecified = false
        do {
            for (key, value) in dictionary {
                switch key {
                case "ko": rules.koRule = try parseKoRule(value as? String ?? "")
                case "score", "scoring": rules.scoringRule = try parseScoringRule(value as? String ?? "")
                case "tax": rules.taxRule = try parseTaxRule(value as? String ?? ""); taxSpecified = true
                case "suicide": guard let bool = value as? Bool
                    else { throw RulesError.couldNotParse(original) }; rules.multiStoneSuicideLegal = bool
                case "hasButton": guard let bool = value as? Bool
                    else { throw RulesError.couldNotParse(original) }; rules.hasButton = bool
                case "whiteHandicapBonus": rules
                    .whiteHandicapBonusRule = try parseWhiteHandicapBonusRule(value as? String ?? "")
                case "friendlyPassOk": guard let bool = value as? Bool
                    else { throw RulesError.couldNotParse(original) }; rules.friendlyPassOk = bool
                case "komi":
                    guard allowKomi, let number = value as? NSNumber else { throw RulesError.unknownOption(key) }
                    rules.komi = number.floatValue
                    komiSpecified = true
                    guard rules.komi >= minUserKomi, rules.komi <= maxUserKomi, komiIsIntOrHalfInt(rules.komi) else {
                        throw RulesError.invalidValue("Komi value is not a half-integer or is too extreme")
                    }
                default: throw RulesError.unknownOption(key)
                }
            }
        } catch { throw error }
        applyInferredDefaults(&rules, taxSpecified: taxSpecified, komiSpecified: komiSpecified)
        return rules
    }

    private static func parseLegacyRules(_ original: String, allowKomi: Bool) throws -> Rules {
        var remaining = original.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !remaining.isEmpty else { throw RulesError.couldNotParse(original) }
        var rules = Rules()
        var komiSpecified = false
        var taxSpecified = false
        func strip(_ prefix: String) -> Bool {
            guard remaining.hasPrefix(prefix) else { return false }
            remaining.removeFirst(prefix.count)
            remaining = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
            return true
        }
        while !remaining.isEmpty {
            if strip("komi") {
                guard allowKomi else { throw RulesError.couldNotParse(original) }
                let end = remaining.firstIndex { $0.isLetter || $0.isWhitespace } ?? remaining.endIndex
                guard let value = Float(remaining[..<end]), value.isFinite, abs(value) <= 100_000 else {
                    throw RulesError.couldNotParse(original)
                }
                rules.komi = value; komiSpecified = true; remaining = String(remaining[end...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else if strip("ko") {
                if strip("SIMPLE") { rules.koRule = .simple } else if strip("POSITIONAL") {
                    rules.koRule = .positional
                } else if strip("SITUATIONAL") {
                    rules.koRule = .situational
                } else if strip("SPIGHT") {
                    rules.koRule = .spight
                } else { throw RulesError.couldNotParse(original) }
            } else if strip("scoring") || strip("score") {
                if strip("AREA") { rules.scoringRule = .area } else if strip("TERRITORY") {
                    rules.scoringRule = .territory
                } else { throw RulesError.couldNotParse(original) }
            } else if strip("tax") {
                if strip("NONE") { rules.taxRule = .none } else if strip("SEKI") {
                    rules.taxRule = .seki
                } else if strip("ALL") {
                    rules.taxRule = .all
                } else { throw RulesError.couldNotParse(original) }
                taxSpecified = true
            } else if strip("sui") {
                if strip("1") { rules.multiStoneSuicideLegal = true } else if strip("0") {
                    rules.multiStoneSuicideLegal = false
                } else { throw RulesError.couldNotParse(original) }
            } else if strip("button") {
                if strip("1") { rules.hasButton = true } else if strip("0") {
                    rules.hasButton = false
                } else { throw RulesError.couldNotParse(original) }
            } else if strip("whb") {
                if strip("0") { rules.whiteHandicapBonusRule = .zero } else if strip("N-1") {
                    rules.whiteHandicapBonusRule = .nMinusOne
                } else if strip("N") {
                    rules.whiteHandicapBonusRule = .n
                } else { throw RulesError.couldNotParse(original) }
            } else if strip("fpok") {
                if strip("1") { rules.friendlyPassOk = true } else if strip("0") {
                    rules.friendlyPassOk = false
                } else { throw RulesError.couldNotParse(original) }
            } else { throw RulesError.couldNotParse(original) }
        }
        applyInferredDefaults(&rules, taxSpecified: taxSpecified, komiSpecified: komiSpecified)
        return rules
    }

    private static func applyInferredDefaults(_ rules: inout Rules, taxSpecified: Bool, komiSpecified: Bool) {
        if !taxSpecified { rules.taxRule = rules.scoringRule == .territory ? .seki : .none }
        if !komiSpecified {
            if rules.scoringRule == .territory { rules.komi = 6.5 } else if rules.hasButton { rules.komi = 7 }
        }
    }

    private func toJsonStringHelper(omitKomi: Bool, omitDefaults: Bool) -> String {
        var fields: [String] = []
        if !omitDefaults || friendlyPassOk { fields.append("\"friendlyPassOk\":\(friendlyPassOk)") }
        if !omitDefaults || hasButton { fields.append("\"hasButton\":\(hasButton)") }
        fields.append("\"ko\":\"\(Self.writeKoRule(koRule))\"")
        if !omitKomi { fields.append("\"komi\":\(Self.formatFloat(komi, forceDecimal: true))") }
        fields.append("\"scoring\":\"\(Self.writeScoringRule(scoringRule))\"")
        fields.append("\"suicide\":\(multiStoneSuicideLegal)")
        fields.append("\"tax\":\"\(Self.writeTaxRule(taxRule))\"")
        if !omitDefaults || whiteHandicapBonusRule != .zero {
            fields.append("\"whiteHandicapBonus\":\"\(Self.writeWhiteHandicapBonusRule(whiteHandicapBonusRule))\"")
        }
        return "{" + fields.joined(separator: ",") + "}"
    }

    private static func parseBool(_ string: String) throws -> Bool {
        if ["1", "TRUE", "YES", "Y", "ON"].contains(string) { return true }
        if ["0", "FALSE", "NO", "N", "OFF"].contains(string) { return false }
        throw RulesError.invalidValue("Invalid boolean: \(string)")
    }

    private static func formatFloat(_ value: Float, forceDecimal: Bool = false) -> String {
        if value.rounded(.towardZero) == value {
            return forceDecimal ? "\(Int(value)).0" : "\(Int(value))"
        }
        return String(format: "%.6g", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
