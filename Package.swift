// swift-tools-version: 6.0
import PackageDescription

// Dependency direction (docs/spec/00-overview.md §3):
//   IchiGoCore ← IchiGoFeatures ← IchiGoEngine ← IchiGoGTP ← ichigo
//   LogicModel (Foundation only) ← LogicMetal (Metal) ; IchiGoEngine → LogicModel
// LogicMetal is the only target that touches a Metal device; every other test target runs
// without a GPU.
let package = Package(
    name: "IchiGo",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "IchiGoCore", targets: ["IchiGoCore"]),
        .library(name: "IchiGoFeatures", targets: ["IchiGoFeatures"]),
        .library(name: "LogicModel", targets: ["LogicModel"]),
        .library(name: "LogicMetal", targets: ["LogicMetal"]),
        .library(name: "IchiGoEngine", targets: ["IchiGoEngine"]),
        .library(name: "IchiGoGTP", targets: ["IchiGoGTP"]),
        .executable(name: "ichigo", targets: ["ichigo"]),
    ],
    targets: [
        .target(name: "IchiGoCore"),
        .target(name: "IchiGoFeatures", dependencies: ["IchiGoCore"]),
        .target(name: "LogicModel"),
        .target(name: "LogicMetal", dependencies: ["LogicModel"]),
        .target(name: "IchiGoEngine", dependencies: ["IchiGoCore", "IchiGoFeatures", "LogicModel"]),
        .target(name: "IchiGoGTP", dependencies: ["IchiGoEngine"]),
        .executableTarget(
            name: "ichigo",
            dependencies: ["IchiGoCore", "IchiGoFeatures", "LogicModel", "LogicMetal", "IchiGoEngine", "IchiGoGTP"]
        ),
        .testTarget(
            name: "IchiGoCoreTests",
            dependencies: ["IchiGoCore"],
            resources: [.copy("Fixtures"), .copy("Goldens")]
        ),
        .testTarget(name: "IchiGoFeaturesTests", dependencies: ["IchiGoFeatures", "IchiGoCore"]),
        .testTarget(name: "LogicModelTests", dependencies: ["LogicModel"]),
        .testTarget(name: "IchiGoEngineTests", dependencies: ["IchiGoEngine"]),
        .testTarget(name: "IchiGoGTPTests", dependencies: ["IchiGoGTP"]),
    ]
)
