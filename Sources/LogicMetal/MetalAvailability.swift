import Foundation
import Metal

/// Reports whether a Metal compute device exists. The Metal gate kernels are T22+; this file
/// exists so `ichigo doctor` can report GPU capability without any other target importing Metal.
public enum MetalAvailability {
    public struct Info: Sendable, Encodable {
        public let available: Bool
        public let deviceName: String?
        public let recommendedMaxWorkingSetBytes: UInt64?
        public let hasUnifiedMemory: Bool?
    }

    public static func probe() -> Info {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return Info(available: false, deviceName: nil, recommendedMaxWorkingSetBytes: nil, hasUnifiedMemory: nil)
        }
        return Info(
            available: true,
            deviceName: device.name,
            recommendedMaxWorkingSetBytes: device.recommendedMaxWorkingSetSize,
            hasUnifiedMemory: device.hasUnifiedMemory
        )
    }
}
