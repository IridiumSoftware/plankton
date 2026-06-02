// All Metal Shading Language for the engine, compiled at runtime.
//
// MIDDLE-ROAD MODEL (2D): agents drive an incompressible fluid.
//   - Particles SENSE the dye field ahead (Physarum/Sage-Jenson) and steer
//     toward it; they swim at a fixed speed along that heading.
//   - They DEPOSIT their steered velocity into the fluid (forcing) + dye.
//   - The velocity field is advected, damped, then PROJECTED divergence-free
//     (Jacobi pressure-Poisson = the discrete Hodge/Helmholtz projection, same
//     operator as navier-stokes' spectral_2d_control.jl, opposite half).
//   - Agents are also CARRIED by the projected fluid (two-way coupling): they
//     drive it, it drifts them. Dye is advected by the flow and rendered, with
//     the agents drawn as points on top.
//
// Velocity stored interleaved (vel[2i]=x, vel[2i+1]=y) in cells/frame; periodic.
enum Shaders {
    static let source = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    struct Particle { float2 pos; float2 vel; };

    // ── Tunable constants (v0) ────────────────────────────────────────────
    constant float VEL_DAMP     = 0.950;   // fluid energy bleed / frame
    constant float DYE_DECAY    = 0.985;   // dye fade / frame
    constant float TONE_K       = 0.020;   // dye → brightness (↓ dimmer/contrast)
    constant float SWIM_SPEED   = 0.10;    // agent self-propulsion
    constant float SENSOR_DIST  = 0.012;   // how far ahead agents sense the dye
    constant float SENSOR_ANGLE = 0.40;    // radians, left/right sensor offset
    constant float TURN         = 0.35;    // radians steered per step
    constant float AGENT_DT     = 0.0167;  // agent step (≈1/60)
    constant float FLUID_PULL   = 2.0;     // how strongly the fluid carries agents
    constant float POINT_SIZE   = 1.5;     // agent dot size (px)
    constant float POINT_ALPHA  = 0.10;    // agent dot brightness (additive)

    // ── helpers ───────────────────────────────────────────────────────────
    static inline int wrapi(int a, int n) { return ((a % n) + n) % n; }
    static inline uint cidx(int x, int y, uint2 d) {
        return uint(wrapi(y, int(d.y))) * d.x + uint(wrapi(x, int(d.x)));
    }
    static inline float2 rotate(float2 v, float a) {
        float c = cos(a), s = sin(a);
        return float2(v.x * c - v.y * s, v.x * s + v.y * c);
    }
    static inline float2 velAt(device const float *v, int x, int y, uint2 d) {
        uint i = cidx(x, y, d); return float2(v[2 * i], v[2 * i + 1]);
    }
    static inline float2 sampleVel(device const float *v, float2 p, uint2 d) {
        float fx = floor(p.x), fy = floor(p.y);
        int x0 = int(fx), y0 = int(fy);
        float tx = p.x - fx, ty = p.y - fy;
        float2 a = mix(velAt(v, x0, y0,     d), velAt(v, x0 + 1, y0,     d), tx);
        float2 b = mix(velAt(v, x0, y0 + 1, d), velAt(v, x0 + 1, y0 + 1, d), tx);
        return mix(a, b, ty);
    }
    static inline float sampleScalar(device const float *s, float2 p, uint2 d) {
        float fx = floor(p.x), fy = floor(p.y);
        int x0 = int(fx), y0 = int(fy);
        float tx = p.x - fx, ty = p.y - fy;
        float a = mix(s[cidx(x0, y0,     d)], s[cidx(x0 + 1, y0,     d)], tx);
        float b = mix(s[cidx(x0, y0 + 1, d)], s[cidx(x0 + 1, y0 + 1, d)], tx);
        return mix(a, b, ty);
    }

    // ── per-element multiply (velocity damp over 2N) ──────────────────────
    kernel void scale_buffer(device float *buf [[buffer(0)]],
                             constant float &k  [[buffer(1)]],
                             uint i             [[thread_position_in_grid]]) {
        buf[i] *= k;
    }

    // ── 1. advect velocity (semi-Lagrangian) ──────────────────────────────
    kernel void advect_velocity(device const float *vin [[buffer(0)]],
                                device float *vout        [[buffer(1)]],
                                constant uint2 &d         [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        uint i = gid.y * d.x + gid.x;
        float2 here = float2(vin[2 * i], vin[2 * i + 1]);
        float2 pos = float2(float(gid.x) + 0.5, float(gid.y) + 0.5) - here;
        float2 sv = sampleVel(vin, pos, d);
        vout[2 * i] = sv.x; vout[2 * i + 1] = sv.y;
    }

    // ── 2. splat: agents deposit steered velocity (forcing) + dye ─────────
    kernel void splat(device atomic_float *vel    [[buffer(0)]],
                      device atomic_float *dye      [[buffer(1)]],
                      device const Particle *parts  [[buffer(2)]],
                      constant uint2 &d            [[buffer(3)]],
                      constant float &forceGain    [[buffer(4)]],
                      constant float &dyeAmt       [[buffer(5)]],
                      uint gid                     [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        float fx = p.pos.x * float(d.x) - 0.5;
        float fy = p.pos.y * float(d.y) - 0.5;
        int x0 = int(floor(fx)), y0 = int(floor(fy));
        float tx = fx - float(x0), ty = fy - float(y0);
        float2 f = p.vel * forceGain;
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int xi = wrapi(x0 + i, int(d.x)), yi = wrapi(y0 + j, int(d.y));
                float w = (i == 0 ? 1.0 - tx : tx) * (j == 0 ? 1.0 - ty : ty);
                uint c = uint(yi) * d.x + uint(xi);
                atomic_fetch_add_explicit(&vel[2 * c],     f.x * w, memory_order_relaxed);
                atomic_fetch_add_explicit(&vel[2 * c + 1], f.y * w, memory_order_relaxed);
                atomic_fetch_add_explicit(&dye[c],         dyeAmt * w, memory_order_relaxed);
            }
        }
    }

    // ── 3. divergence ─────────────────────────────────────────────────────
    kernel void divergence(device const float *v [[buffer(0)]],
                           device float *divg      [[buffer(1)]],
                           constant uint2 &d       [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        int x = int(gid.x), y = int(gid.y);
        float vl = velAt(v, x - 1, y, d).x, vr = velAt(v, x + 1, y, d).x;
        float vb = velAt(v, x, y - 1, d).y, vt = velAt(v, x, y + 1, d).y;
        divg[gid.y * d.x + gid.x] = 0.5 * ((vr - vl) + (vt - vb));
    }

    // ── 4. Jacobi iteration of ∇²p = div ──────────────────────────────────
    kernel void jacobi(device const float *pin [[buffer(0)]],
                       device float *pout        [[buffer(1)]],
                       device const float *divg  [[buffer(2)]],
                       constant uint2 &d         [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        int x = int(gid.x), y = int(gid.y);
        float pl = pin[cidx(x - 1, y, d)], pr = pin[cidx(x + 1, y, d)];
        float pb = pin[cidx(x, y - 1, d)], pt = pin[cidx(x, y + 1, d)];
        pout[gid.y * d.x + gid.x] = (pl + pr + pb + pt - divg[gid.y * d.x + gid.x]) * 0.25;
    }

    // ── 5. subtract pressure gradient → divergence-free ───────────────────
    kernel void subtract_gradient(device float *v     [[buffer(0)]],
                                  device const float *p [[buffer(1)]],
                                  constant uint2 &d     [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        int x = int(gid.x), y = int(gid.y);
        float pl = p[cidx(x - 1, y, d)], pr = p[cidx(x + 1, y, d)];
        float pb = p[cidx(x, y - 1, d)], pt = p[cidx(x, y + 1, d)];
        uint i = gid.y * d.x + gid.x;
        v[2 * i]     -= 0.5 * (pr - pl);
        v[2 * i + 1] -= 0.5 * (pt - pb);
    }

    // ── 6. advect dye by the projected field (+ decay) ────────────────────
    kernel void advect_dye(device const float *din [[buffer(0)]],
                           device float *dout        [[buffer(1)]],
                           device const float *v     [[buffer(2)]],
                           constant uint2 &d         [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        uint i = gid.y * d.x + gid.x;
        float2 vel = float2(v[2 * i], v[2 * i + 1]);
        float2 pos = float2(float(gid.x) + 0.5, float(gid.y) + 0.5) - vel;
        dout[i] = sampleScalar(din, pos, d) * DYE_DECAY;
    }

    // ── 7. agents: sense dye → steer → swim + carried by the fluid ────────
    kernel void move_particles(device Particle *parts   [[buffer(0)]],
                               device const float *v     [[buffer(1)]],
                               device const float *dye   [[buffer(2)]],
                               constant uint2 &d         [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        float2 dd = float2(d);
        float spd = length(p.vel);
        float2 dir = spd > 1e-6 ? p.vel / spd : float2(1.0, 0.0);

        // sense the dye ahead (left / center / right) and steer toward it
        float cF = sampleScalar(dye, (p.pos + dir * SENSOR_DIST) * dd, d);
        float lF = sampleScalar(dye, (p.pos + rotate(dir,  SENSOR_ANGLE) * SENSOR_DIST) * dd, d);
        float rF = sampleScalar(dye, (p.pos + rotate(dir, -SENSOR_ANGLE) * SENSOR_DIST) * dd, d);
        float steer = (cF >= lF && cF >= rF) ? 0.0 : (lF > rF ? TURN : -TURN);
        dir = rotate(dir, steer);
        p.vel = dir * SWIM_SPEED;                 // steered swim velocity = the fluid forcing

        // self-propulsion + advection by the projected fluid
        float2 fv = sampleVel(v, p.pos * dd, d);  // cells/frame
        float2 disp = dir * SWIM_SPEED * AGENT_DT + (fv / dd) * FLUID_PULL;
        p.pos = fract(p.pos + disp);
        parts[gid] = p;
    }

    // ── fullscreen render: dye → calm glow ────────────────────────────────
    struct VSOut { float4 position [[position]]; float2 uv; };
    vertex VSOut fs_vertex(uint vid [[vertex_id]]) {
        float2 uv = float2((vid << 1) & 2, vid & 2);
        VSOut o;
        o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
        o.uv = uv;
        return o;
    }
    fragment float4 fs_fragment(VSOut in                 [[stage_in]],
                                device const float *dye   [[buffer(0)]],
                                constant uint2 &d         [[buffer(1)]]) {
        int x = clamp(int(in.uv.x * float(d.x)), 0, int(d.x) - 1);
        int y = clamp(int(in.uv.y * float(d.y)), 0, int(d.y) - 1);
        float v = dye[uint(y) * d.x + uint(x)];
        float a = 1.0 - exp(-v * TONE_K);
        float3 col = mix(float3(0.02, 0.03, 0.07), float3(0.20, 0.85, 0.95), a);
        return float4(col, 1.0);
    }

    // ── agent points (drawn additively over the dye) ──────────────────────
    struct PointOut { float4 position [[position]]; float pointSize [[point_size]]; };
    vertex PointOut point_vertex(uint vid                  [[vertex_id]],
                                 device const Particle *parts [[buffer(0)]]) {
        Particle p = parts[vid];
        PointOut o;
        o.position = float4(p.pos * 2.0 - 1.0, 0.0, 1.0);
        o.pointSize = POINT_SIZE;
        return o;
    }
    fragment float4 point_fragment(float2 pc [[point_coord]]) {
        if (length(pc - 0.5) > 0.5) discard_fragment();
        return float4(float3(0.7, 0.95, 1.0) * POINT_ALPHA, POINT_ALPHA);
    }
    """
}
