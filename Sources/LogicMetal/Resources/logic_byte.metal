#include <metal_stdlib>
using namespace metal;

// Metal-byte gate kernel (docs/spec/01-network.md §5, T22). Same UInt8 `[B,S,S,C]` activation
// layout as `ScalarBackend` (`(((b*S+y)*S+x)*C+c)`), one dispatch per logic layer. The grid is
// (x, y, b*C+c); each thread evaluates exactly one output bit, matching `ScalarBackend.read`:
//   - off-board reads (x+dx, y+dy outside [0,S)) return 0, no wraparound.
//   - bank 1 reads the original input tensor (`inputTensor`/`inputC`).
//   - bank 0 reads the previous layer's output (`prevTensor`/`prevC`); for layer 0 the caller
//     passes `prevTensor == inputTensor` and `prevC == inputC`, since layer 0's "previous layer"
//     is the 32-channel input.
//
// Descriptors are plain Int32/UInt32 arrays, never a raw struct laid across the Swift/Metal
// boundary:
//   - `wiring` is `[C][2][4]` flattened int32: (bank, channel, dx, dy) for reference A then B.
//   - `gates` is `[C]` uint8 truth-table ids (0...15), encoding `(g >> (2*a+b)) & 1`.
//   - `dims` is `[B, S, C, inputChannels, prevChannels]` int32.
kernel void logic_layer(
    device const uchar* inputTensor [[buffer(0)]],
    device const uchar* prevTensor  [[buffer(1)]],
    device uchar* outTensor         [[buffer(2)]],
    device const int* wiring        [[buffer(3)]],
    device const uchar* gates       [[buffer(4)]],
    device const int* dims          [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int B = dims[0];
    const int S = dims[1];
    const int C = dims[2];
    const int inputC = dims[3];
    const int prevC = dims[4];

    const int x = int(gid.x);
    const int y = int(gid.y);
    const int bc = int(gid.z);
    if (x >= S || y >= S || bc >= B * C) {
        return;
    }
    const int b = bc / C;
    const int c = bc % C;

    uchar bits[2];
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
            const int idx = ((b * S + yy) * S + xx) * inputC + channel;
            bits[k] = inputTensor[idx];
        } else {
            const int idx = ((b * S + yy) * S + xx) * prevC + channel;
            bits[k] = prevTensor[idx];
        }
    }
    const int row = 2 * int(bits[0]) + int(bits[1]);
    const uchar g = gates[c];
    const uchar outBit = (g >> row) & 1;
    const int outIdx = ((b * S + y) * S + x) * C + c;
    outTensor[outIdx] = outBit;
}
