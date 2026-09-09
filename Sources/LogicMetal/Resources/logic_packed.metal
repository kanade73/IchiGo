#include <metal_stdlib>
using namespace metal;

// Metal-packed gate kernel (docs/spec/01-network.md §5, T23). Batch is packed 32-at-a-time into
// `uint` words: `packed[((group*S+y)*S+x)*C+c]` bit k is sample `b = group*32+k`.
// `G = ceil(B/32)`. One dispatch per logic layer, mirroring `logic_byte.metal`'s structure
// exactly except the payload type is `uint` (32 lanes/word) instead of `uchar` (1 lane/word), and
// every output word is masked by that group's valid-lane mask before it is written -- including
// gate 15 (true) and gate 3 (NOT a), which would otherwise set the padding lanes of a partial
// last group to 1 (off-board/zero-padded `a`/`b` reads make both of those gates evaluate to 1
// unconditionally).
//
// Descriptors are plain Int32/UInt32 arrays, never a raw struct laid across the Swift/Metal
// boundary:
//   - `wiring` is `[C][2][4]` flattened int32: (bank, channel, dx, dy) for reference A then B.
//   - `gates` is `[C]` uint8 truth-table ids (0...15).
//   - `dims` is `[G, S, C, inputChannels, prevChannels, B]` int32.
kernel void logic_layer_packed(
    device const uint* inputTensor  [[buffer(0)]],
    device const uint* prevTensor   [[buffer(1)]],
    device uint* outTensor          [[buffer(2)]],
    device const int* wiring        [[buffer(3)]],
    device const uchar* gates       [[buffer(4)]],
    device const int* dims          [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int G = dims[0];
    const int S = dims[1];
    const int C = dims[2];
    const int inputC = dims[3];
    const int prevC = dims[4];
    const int B = dims[5];

    const int x = int(gid.x);
    const int y = int(gid.y);
    const int gc = int(gid.z);
    if (x >= S || y >= S || gc >= G * C) {
        return;
    }
    const int group = gc / C;
    const int c = gc % C;

    uint bits[2];
    for (int k = 0; k < 2; k++) {
        const int base = (c * 2 + k) * 4;
        const int bank = wiring[base + 0];
        const int channel = wiring[base + 1];
        const int dx = wiring[base + 2];
        const int dy = wiring[base + 3];
        const int xx = x + dx;
        const int yy = y + dy;
        if (xx < 0 || xx >= S || yy < 0 || yy >= S) {
            bits[k] = 0;
            continue;
        }
        if (bank == 1) {
            const int idx = ((group * S + yy) * S + xx) * inputC + channel;
            bits[k] = inputTensor[idx];
        } else {
            const int idx = ((group * S + yy) * S + xx) * prevC + channel;
            bits[k] = prevTensor[idx];
        }
    }
    const uint a = bits[0];
    const uint b = bits[1];
    const uchar g = gates[c];
    uint out = 0;
    if (g & 1) out |= (~a & ~b);
    if (g & 2) out |= (~a & b);
    if (g & 4) out |= (a & ~b);
    if (g & 8) out |= (a & b);

    uint valid;
    if (group == G - 1) {
        const int remainder = B % 32;
        // Never shift by 32 ("シフト32は禁止"): remainder is in [1,31] whenever it's nonzero.
        valid = (remainder == 0) ? 0xffffffffu : ((1u << uint(remainder)) - 1u);
    } else {
        valid = 0xffffffffu;
    }
    out &= valid;

    const int outIdx = ((group * S + y) * S + x) * C + c;
    outTensor[outIdx] = out;
}
