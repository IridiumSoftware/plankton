// All Metal Shading Language for the engine, compiled at runtime.
//
// FAITHFUL MODEL (2D): active agents drive a faithful incompressible NS fluid.
//   - Particles SENSE the flow + the dye (density) ahead, steer by the Fourier
//     brain + chemotaxis (toward higher dye = aggregation), and FORCE the fluid
//     as a NET-ZERO force DIPOLE (+f ahead, −f behind — a real active swimmer).
//   - The velocity field is advected, given real ν∇² VISCOSITY (scale-selective,
//     `diffuse_velocity`) + a weak large-scale drag, then PROJECTED divergence-free
//     (Jacobi pressure-Poisson = discrete Hodge/Helmholtz projection).
//   - Agents are also carried by the projected fluid (two-way). Dye is advected
//     by the flow and rendered; agents drawn as additive points on top.
//
// FAITHFUL RETROFIT (2026-06-05): the two unphysical pieces of the original engine
// — uniform drag (no inertial range) and a momentum-monopole splat (net force ≠ 0)
// — are replaced by ν∇² viscosity and a net-zero force dipole, the fixes validated
// in the navier-stokes repo (AT-1..6 / SIM_SPEC.md). The fluid now behaves like real
// 2D NS (a universal −3 enstrophy cascade); the lifelike creatures come from the
// chemotaxis (`cohesion`), which survives on the faithful fluid (AT-5). See
// docs/faithful_fluid.md.
//
// Tunables are now UNIFORMS (see Params.swift / Tuning.swift) so they can be
// dialed live. Velocity is interleaved (vel[2i]=x, vel[2i+1]=y), cells/frame,
// periodic domain.
enum Shaders {
    static let source = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    struct Particle { float2 pos; float2 vel; };
    struct MoveParams { float swim; float sensorDist; float sensorAngle; float turn; float fluidPull; float senseScale; float speedGain; float cohesion; };
    struct MouseUniform { float2 pos; float2 vel; float radius; float forceGain; float dyeGain; float active; };

    // fixed (non-live) constants
    constant float AGENT_DT = 0.0167;     // agent step (≈1/60)

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

    // ── 1b. viscous diffusion ν∇²v (explicit 5-point Laplacian) — the FAITHFUL
    //        fix for fluoddity's uniform drag. Real viscosity is k²-SELECTIVE
    //        (kills small scales, spares large ⇒ an inertial range / cascade);
    //        uniform drag damped every mode equally ⇒ no cascade (the "dial").
    //        Stable for visc ≤ 0.25 (explicit-diffusion limit). Mirrors the
    //        navier-stokes AT-1/AT-6 ν∇² fix, grid-discretized for live stepping.
    kernel void diffuse_velocity(device const float *vin [[buffer(0)]],
                                 device float *vout        [[buffer(1)]],
                                 constant uint2 &d         [[buffer(2)]],
                                 constant float &visc      [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        int x = int(gid.x), y = int(gid.y);
        uint i = gid.y * d.x + gid.x;
        float2 c = float2(vin[2 * i], vin[2 * i + 1]);
        float2 lap = velAt(vin, x - 1, y, d) + velAt(vin, x + 1, y, d)
                   + velAt(vin, x, y - 1, d) + velAt(vin, x, y + 1, d) - 4.0 * c;
        float2 o = c + visc * lap;
        vout[2 * i] = o.x; vout[2 * i + 1] = o.y;
    }

    // bilinear force deposit at a grid point (cells) — used twice for the dipole
    static inline void depositForce(device atomic_float *vel, float fx, float fy,
                                    float2 f, uint2 d) {
        int x0 = int(floor(fx)), y0 = int(floor(fy));
        float tx = fx - float(x0), ty = fy - float(y0);
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int xi = wrapi(x0 + i, int(d.x)), yi = wrapi(y0 + j, int(d.y));
                float w = (i == 0 ? 1.0 - tx : tx) * (j == 0 ? 1.0 - ty : ty);
                uint c = uint(yi) * d.x + uint(xi);
                atomic_fetch_add_explicit(&vel[2 * c],     f.x * w, memory_order_relaxed);
                atomic_fetch_add_explicit(&vel[2 * c + 1], f.y * w, memory_order_relaxed);
            }
        }
    }

    // ── 2. splat: agents force the fluid as a NET-ZERO force DIPOLE (+f at the
    //        head, −f at the tail ⇒ Σmomentum = 0 — the physical active-swimmer
    //        forcing, the FAITHFUL fix for the momentum-monopole splat) and
    //        deposit dye (= local agent density, the chemotaxis signal) at center.
    kernel void splat(device atomic_float *vel    [[buffer(0)]],
                      device atomic_float *dye      [[buffer(1)]],
                      device const Particle *parts  [[buffer(2)]],
                      constant uint2 &d            [[buffer(3)]],
                      constant float &forceGain    [[buffer(4)]],
                      constant float &dyeAmt       [[buffer(5)]],
                      constant float &dipoleLen    [[buffer(6)]],
                      uint gid                     [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        float cx = p.pos.x * float(d.x) - 0.5;
        float cy = p.pos.y * float(d.y) - 0.5;
        float spd = length(p.vel);
        float2 fwd = spd > 1e-6 ? p.vel / spd : float2(1.0, 0.0);
        float2 f = p.vel * forceGain;
        float2 h = fwd * (dipoleLen * 0.5);
        depositForce(vel, cx + h.x, cy + h.y,  f, d);   // +f ahead (head)
        depositForce(vel, cx - h.x, cy - h.y, -f, d);   // −f behind (tail) ⇒ net-zero
        // dye at the agent center (bilinear) — density marker the chemotaxis reads
        int x0 = int(floor(cx)), y0 = int(floor(cy));
        float tx = cx - float(x0), ty = cy - float(y0);
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int xi = wrapi(x0 + i, int(d.x)), yi = wrapi(y0 + j, int(d.y));
                float w = (i == 0 ? 1.0 - tx : tx) * (j == 0 ? 1.0 - ty : ty);
                atomic_fetch_add_explicit(&dye[uint(yi) * d.x + uint(xi)], dyeAmt * w, memory_order_relaxed);
            }
        }
    }

    // ── 2b. mouse stir: add a Gaussian blob of velocity + dye at the cursor ─
    kernel void mouse_stir(device float *vel       [[buffer(0)]],
                           device float *dye         [[buffer(1)]],
                           constant uint2 &d         [[buffer(2)]],
                           constant MouseUniform &m  [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        float2 cell = float2(float(gid.x) + 0.5, float(gid.y) + 0.5);
        float2 mc = m.pos * float2(d);
        float2 diff = (cell - mc) / max(m.radius * float(d.x), 1.0);
        float g = exp(-dot(diff, diff));
        if (g < 0.003) return;
        uint i = gid.y * d.x + gid.x;
        vel[2 * i]     += m.vel.x * float(d.x) * m.forceGain * g;
        vel[2 * i + 1] += m.vel.y * float(d.x) * m.forceGain * g;
        dye[i]         += m.dyeGain * g;
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
                           constant float &dyeDecay  [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        uint i = gid.y * d.x + gid.x;
        float2 vel = float2(v[2 * i], v[2 * i + 1]);
        float2 pos = float2(float(gid.x) + 0.5, float(gid.y) + 0.5) - vel;
        dout[i] = sampleScalar(din, pos, d) * dyeDecay;
    }

    // ── 6b. separable box blur (run twice: horizontal then vertical) for the
    //        bloom glow buffer ─────────────────────────────────────────────
    kernel void box_blur(device const float *src [[buffer(0)]],
                         device float *dst         [[buffer(1)]],
                         constant uint2 &d         [[buffer(2)]],
                         constant int2 &step       [[buffer(3)]],
                         constant int &radius      [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        float sum = 0.0;
        for (int k = -radius; k <= radius; k++) {
            sum += src[cidx(int(gid.x) + step.x * k, int(gid.y) + step.y * k, d)];
        }
        dst[gid.y * d.x + gid.x] = sum / float(2 * radius + 1);
    }

    // 80-param symmetric brain: 10 Fourier centers (rule[0..9] = frequencies,
    // rule[10..19] = amplitudes), a 4D→4D sum of sines.  out = Σ amp·sin(freq·x).
    constant int N_COHORTS = 8;   // sub-populations, each with its own 80-param rule

    // `off` selects the cohort's block (cohort * 20 float4 = freqs[10] + amps[10]).
    static inline float4 fourier(device const float4 *rule, uint off, float4 x) {
        float4 acc = float4(0.0);
        for (int i = 0; i < 10; i++) {
            acc += rule[off + 10 + i] * sin(dot(rule[off + i], x));
        }
        return acc;
    }

    // ── 7. agents: sense the flow at two sensors → Fourier brain (mirror-
    //        averaged) → steer/speed → swim + carried by the fluid ──────────
    kernel void move_particles(device Particle *parts      [[buffer(0)]],
                               device const float *v        [[buffer(1)]],
                               constant uint2 &d            [[buffer(2)]],
                               constant MoveParams &mp      [[buffer(3)]],
                               device const float4 *rule    [[buffer(4)]],
                               constant uint &count         [[buffer(5)]],
                               device const float *dye      [[buffer(6)]],
                               uint gid [[thread_position_in_grid]]) {
        Particle p = parts[gid];
        uint off = ((gid * uint(N_COHORTS)) / count) * 20;   // this agent's cohort block
        float2 dd = float2(d);
        float spd = length(p.vel);
        float2 fwd = spd > 1e-6 ? p.vel / spd : float2(1.0, 0.0);
        float2 lft = float2(fwd.y, -fwd.x);

        // sample the velocity field at the left/right sensors, in the agent's
        // local (forward, left) frame → rotation-invariant brain input
        float2 vL = sampleVel(v, (p.pos + rotate(fwd,  mp.sensorAngle) * mp.sensorDist) * dd, d);
        float2 vR = sampleVel(v, (p.pos + rotate(fwd, -mp.sensorAngle) * mp.sensorDist) * dd, d);
        float2 L = float2(dot(vL, fwd), dot(vL, lft)) * mp.senseScale;
        float2 R = float2(dot(vR, fwd), dot(vR, lft)) * mp.senseScale;

        // evaluate twice, mirrored, and combine: axial reflection-even,
        // turn reflection-odd (keeps the rule free of clockwise/ccw bias)
        float4 base = fourier(rule, off, float4(L.x, L.y, R.x, R.y));
        float4 mir  = fourier(rule, off, float4(R.x, -R.y, L.x, -L.y));
        float axial = base.x + mir.x;
        float turnT = base.y - mir.y;

        // chemotaxis: steer toward higher dye (= agent density) → aggregation,
        // the ingredient that lets agents clump into membrane/vacuole structures
        float dL = sampleScalar(dye, (p.pos + rotate(fwd,  mp.sensorAngle) * mp.sensorDist) * dd, d);
        float dR = sampleScalar(dye, (p.pos + rotate(fwd, -mp.sensorAngle) * mp.sensorDist) * dd, d);
        float chemTurn = (dL - dR) * mp.cohesion;

        float2 nd = rotate(fwd, turnT * mp.turn + chemTurn);
        float newSpeed = clamp(mp.swim * (1.0 + axial * mp.speedGain),
                               mp.swim * 0.2, mp.swim * 2.0);
        p.vel = nd * newSpeed;                     // brain-steered velocity = fluid forcing

        float2 fv = sampleVel(v, p.pos * dd, d);
        float2 disp = nd * newSpeed * AGENT_DT + (fv / dd) * mp.fluidPull;
        p.pos = fract(p.pos + disp);
        parts[gid] = p;
    }

    // ── research-viz: vorticity ω = ∂v/∂x − ∂u/∂y (the 2D-NS state variable) ─
    kernel void vorticity(device const float *v [[buffer(0)]],
                          device float *vort      [[buffer(1)]],
                          constant uint2 &d       [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y) return;
        int x = int(gid.x), y = int(gid.y);
        float vyR = velAt(v, x + 1, y, d).y, vyL = velAt(v, x - 1, y, d).y;
        float vxT = velAt(v, x, y + 1, d).x, vxB = velAt(v, x, y - 1, d).x;
        vort[gid.y * d.x + gid.x] = 0.5 * ((vyR - vyL) - (vxT - vxB));
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
    static inline float3 hsv2rgb(float3 c) {
        float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
    fragment float4 fs_fragment(VSOut in                     [[stage_in]],
                                device const float *dye       [[buffer(0)]],
                                constant uint2 &d             [[buffer(1)]],
                                constant float &toneK         [[buffer(2)]],
                                device const float *vfield    [[buffer(3)]],
                                constant float &satGain       [[buffer(4)]],
                                device const float *dyeBlur   [[buffer(5)]],
                                constant float &bloomStrength [[buffer(6)]],
                                constant float &palette       [[buffer(7)]],
                                device const float *vort      [[buffer(8)]],
                                constant float &viewMode      [[buffer(9)]],
                                constant float &vortScale     [[buffer(10)]],
                                device const float *divField  [[buffer(11)]]) {
        int x = clamp(int(in.uv.x * float(d.x)), 0, int(d.x) - 1);
        int y = clamp(int(in.uv.y * float(d.y)), 0, int(d.y) - 1);
        uint idx = uint(y) * d.x + uint(x);

        // research-viz overlays
        int vm = int(viewMode + 0.5);
        if (vm == 1) {                 // vorticity ω — CCW warm, CW cool
            float w = vort[idx] * vortScale;
            float3 col = (w >= 0.0)
                ? mix(float3(0.03, 0.03, 0.05), float3(1.0, 0.45, 0.1), clamp(w, 0.0, 1.0))
                : mix(float3(0.03, 0.03, 0.05), float3(0.1, 0.5, 1.0), clamp(-w, 0.0, 1.0));
            return float4(col, 1.0);
        }
        if (vm == 2) {                 // enstrophy |ω|² — rotational intensity
            float e = vort[idx] * vort[idx] * vortScale * vortScale;
            return float4(mix(float3(0.0, 0.0, 0.08), float3(1.0, 0.95, 0.4),
                              1.0 - exp(-e)), 1.0);
        }
        if (vm == 3) {                 // divergence — should be ≈0 (incompressible)
            float dv = divField[idx] * vortScale * 8.0;
            float3 col = (dv >= 0.0)
                ? mix(float3(0.02, 0.02, 0.04), float3(1.0, 0.3, 0.3), clamp(dv, 0.0, 1.0))
                : mix(float3(0.02, 0.02, 0.04), float3(0.3, 0.5, 1.0), clamp(-dv, 0.0, 1.0));
            return float4(col, 1.0);
        }

        // vm == 0: the dye art
        float bright = 1.0 - exp(-dye[idx] * toneK);
        float2 fv = float2(vfield[2 * idx], vfield[2 * idx + 1]);
        float speed = length(fv);
        int pal = int(palette + 0.5);

        float3 col;
        if (pal <= 0) {
            // 0: hue = flow direction, saturation = speed
            float hue = atan2(fv.y, fv.x) / 6.283185 + 0.5;
            col = hsv2rgb(float3(hue, clamp(speed * satGain, 0.0, 1.0), bright));
        } else if (pal == 1) {
            // 1: thermal — speed ramps blue → cyan → white → amber
            float t = clamp(speed * satGain, 0.0, 1.0);
            float3 c = mix(float3(0.0, 0.05, 0.25), float3(0.0, 0.7, 0.9), smoothstep(0.0, 0.5, t));
            c = mix(c, float3(1.0, 1.0, 0.85), smoothstep(0.4, 0.8, t));
            c = mix(c, float3(1.0, 0.55, 0.1), smoothstep(0.75, 1.0, t));
            col = c * bright;
        } else {
            // 2: teal duotone (calm)
            col = mix(float3(0.02, 0.04, 0.06), float3(0.10, 0.90, 0.80), bright);
        }

        // bloom: soft glow from the blurred dye
        float glow = 1.0 - exp(-dyeBlur[idx] * toneK);
        col += float3(0.35, 0.7, 0.95) * glow * bloomStrength;
        return float4(col, 1.0);
    }

    // ── agent points: additive, sized by speed, hued by heading ───────────
    struct PointOut { float4 position [[position]]; float pointSize [[point_size]]; float hue; };
    vertex PointOut point_vertex(uint vid                  [[vertex_id]],
                                 device const Particle *parts [[buffer(0)]],
                                 constant float &pointSize    [[buffer(1)]],
                                 constant uint &count         [[buffer(2)]]) {
        Particle p = parts[vid];
        float spd = length(p.vel);
        uint cohort = (vid * uint(N_COHORTS)) / count;
        PointOut o;
        o.position = float4(p.pos * 2.0 - 1.0, 0.0, 1.0);
        o.pointSize = pointSize * (1.0 + spd * 4.0);
        o.hue = float(cohort) / float(N_COHORTS);          // tint by cohort
        return o;
    }
    fragment float4 point_fragment(PointOut in            [[stage_in]],
                                   float2 pc             [[point_coord]],
                                   constant float &alpha  [[buffer(0)]]) {
        float r = length(pc - 0.5) * 2.0;          // 0 center → 1 edge
        if (r > 1.0) discard_fragment();
        float a = alpha * (1.0 - r * r);           // soft round falloff
        return float4(hsv2rgb(float3(in.hue, 0.55, 1.0)) * a, a);
    }
    """
}
