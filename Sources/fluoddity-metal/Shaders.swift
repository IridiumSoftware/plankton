// All Metal Shading Language for the engine, compiled at runtime
// (device.makeLibrary(source:)). Kept in one string so both the Simulation
// (compute) and the Renderer (draw) pull functions from a single library.
//
// v0 model (2D): particles sense the trail field at three points ahead, steer
// toward the strongest (handed Physarum/Sage-Jenson rule), and atomic-splat a
// trail; the field decays each frame. The symmetric Fourier "brain" and the
// advect→diffuse→project fluid step come in later increments.
enum Shaders {
    static let source = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    struct Particle {
        float2 pos;   // normalized [0,1)
        float2 vel;   // velocity (normalized units / sec); |vel| is the speed
    };

    // ── Tunable constants (v0; will migrate to uniforms) ──────────────────
    constant float SENSOR_DIST  = 0.012;   // how far ahead particles look
    constant float SENSOR_ANGLE = 0.40;    // radians, left/right sensor offset
    constant float TURN         = 0.30;    // radians steered per step
    constant float TONE_K       = 0.020;   // field→brightness saturation rate
                                           //   ↑ brighter / washed-out, ↓ dimmer / more contrast

    // ── helpers ───────────────────────────────────────────────────────────
    static inline float2 rotate(float2 v, float a) {
        float c = cos(a), s = sin(a);
        return float2(v.x * c - v.y * s, v.x * s + v.y * c);
    }
    static inline float sample_field(device const float *field, uint2 dim, float2 uv) {
        uv = fract(uv);
        int x = clamp(int(uv.x * float(dim.x)), 0, int(dim.x) - 1);
        int y = clamp(int(uv.y * float(dim.y)), 0, int(dim.y) - 1);
        return field[uint(y) * dim.x + uint(x)];
    }

    // ── decay: fade the trail field in place ──────────────────────────────
    kernel void decay_field(device float   *field  [[buffer(0)]],
                            constant float &decay   [[buffer(1)]],
                            uint gid                [[thread_position_in_grid]]) {
        field[gid] *= decay;
    }

    // ── splat: each particle deposits into the field (bilinear, atomic) ───
    kernel void splat_particles(device atomic_float *field      [[buffer(0)]],
                                device const Particle *parts     [[buffer(1)]],
                                constant uint2 &dim              [[buffer(2)]],
                                constant float &amount           [[buffer(3)]],
                                uint gid                         [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        float fx = p.pos.x * float(dim.x) - 0.5;
        float fy = p.pos.y * float(dim.y) - 0.5;
        int x0 = int(floor(fx)), y0 = int(floor(fy));
        float tx = fx - float(x0), ty = fy - float(y0);
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int xi = ((x0 + i) % int(dim.x) + int(dim.x)) % int(dim.x);
                int yi = ((y0 + j) % int(dim.y) + int(dim.y)) % int(dim.y);
                float w = (i == 0 ? (1.0 - tx) : tx) * (j == 0 ? (1.0 - ty) : ty);
                uint idx = uint(yi) * dim.x + uint(xi);
                atomic_fetch_add_explicit(&field[idx], amount * w, memory_order_relaxed);
            }
        }
    }

    // ── move: sense three points ahead, steer, step + wrap ────────────────
    kernel void move_particles(device Particle *parts        [[buffer(0)]],
                               device const float *field     [[buffer(1)]],
                               constant uint2 &dim           [[buffer(2)]],
                               constant float &dt            [[buffer(3)]],
                               uint gid                      [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        float speed = length(p.vel);
        float2 dir = speed > 1e-6 ? p.vel / speed : float2(1.0, 0.0);

        float cF = sample_field(field, dim, p.pos + dir * SENSOR_DIST);
        float lF = sample_field(field, dim, p.pos + rotate(dir,  SENSOR_ANGLE) * SENSOR_DIST);
        float rF = sample_field(field, dim, p.pos + rotate(dir, -SENSOR_ANGLE) * SENSOR_DIST);

        float steer = 0.0;
        if (cF >= lF && cF >= rF)      steer = 0.0;
        else if (lF > rF)              steer = TURN;
        else                           steer = -TURN;

        float2 nd = rotate(dir, steer);
        p.vel = nd * speed;
        p.pos = fract(p.pos + p.vel * dt);
        parts[gid] = p;
    }

    // ── fullscreen render: sample field → calm glow ───────────────────────
    struct VSOut { float4 position [[position]]; float2 uv; };

    vertex VSOut fs_vertex(uint vid [[vertex_id]]) {
        float2 uv = float2((vid << 1) & 2, vid & 2);  // (0,0),(2,0),(0,2)
        VSOut o;
        o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
        o.uv = uv;
        return o;
    }

    fragment float4 fs_fragment(VSOut in                   [[stage_in]],
                                device const float *field  [[buffer(0)]],
                                constant uint2 &dim        [[buffer(1)]]) {
        int x = clamp(int(in.uv.x * float(dim.x)), 0, int(dim.x) - 1);
        int y = clamp(int(in.uv.y * float(dim.y)), 0, int(dim.y) - 1);
        float v = field[uint(y) * dim.x + uint(x)];
        float a = 1.0 - exp(-v * TONE_K);                 // saturating brightness
        float3 col = mix(float3(0.02, 0.03, 0.07), float3(0.20, 0.85, 0.95), a);
        return float4(col, 1.0);
    }
    """
}
