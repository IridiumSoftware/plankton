// 3D engine shaders.
// Stage 2a: a real 3D incompressible fluid on a 128³ grid — agents splat their
// velocity (forcing), the field is advected (semi-Lagrangian) + damped, then
// PROJECTED divergence-free via a 6-neighbour Jacobi pressure solve (the discrete
// Hodge projection in 3D). Agents ride the projected field. Rendered as an
// additive point cloud through the orbital camera.
// Stage 2b adds the Monte-Carlo tangent-plane Fourier brain.
//
// Velocity stored interleaved (vel[3i + {0,1,2}]) in cells/frame; periodic cube.
enum Shaders3D {
    static let source = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    struct Particle3D { float3 pos; float3 vel; };
    struct MoveParams3 { float swim; float sensorDist; float sensorAngle; float turn; float axialForce; float fluidPull; float senseScale; float planeSamples; float cohesion; };
    constant float AGENT_DT3 = 0.0167;

    static inline int wrp(int a, int n) { return ((a % n) + n) % n; }
    static inline uint idx3(int x, int y, int z, uint3 d) {
        return (uint(wrp(z, int(d.z))) * d.y + uint(wrp(y, int(d.y)))) * d.x + uint(wrp(x, int(d.x)));
    }
    static inline float3 vat(device const float *v, int x, int y, int z, uint3 d) {
        uint i = idx3(x, y, z, d); return float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    }
    static inline float3 sampleVel3(device const float *v, float3 p, uint3 d) {
        float3 fl = floor(p); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z);
        float3 t = p - fl;
        float3 c00 = mix(vat(v, x0, y0, z0, d),     vat(v, x0 + 1, y0, z0, d),     t.x);
        float3 c10 = mix(vat(v, x0, y0 + 1, z0, d), vat(v, x0 + 1, y0 + 1, z0, d), t.x);
        float3 c01 = mix(vat(v, x0, y0, z0 + 1, d), vat(v, x0 + 1, y0, z0 + 1, d), t.x);
        float3 c11 = mix(vat(v, x0, y0 + 1, z0 + 1, d), vat(v, x0 + 1, y0 + 1, z0 + 1, d), t.x);
        return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
    }
    static inline float hashf(float x) { return fract(sin(x) * 43758.5453); }
    // 80-param symmetric brain (2D): 10 Fourier centers, 4D→4D sum of sines.
    // 8 cohorts (mirrors the 2D engine): `off` selects this agent's 20-float4
    // block (freqs[10] + amps[10]) — the substrate for right-click breeding.
    constant int N_COHORTS3 = 8;
    static inline float4 fourier3(device const float4 *rule, uint off, float4 x) {
        float4 acc = float4(0.0);
        for (int i = 0; i < 10; i++) acc += rule[off + 10 + i] * sin(dot(rule[off + i], x));
        return acc;
    }
    static inline float sampleScalar3(device const float *s, float3 p, uint3 d) {
        float3 fl = floor(p); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z); float3 t = p - fl;
        float c00 = mix(s[idx3(x0,y0,z0,d)],     s[idx3(x0+1,y0,z0,d)],     t.x);
        float c10 = mix(s[idx3(x0,y0+1,z0,d)],   s[idx3(x0+1,y0+1,z0,d)],   t.x);
        float c01 = mix(s[idx3(x0,y0,z0+1,d)],   s[idx3(x0+1,y0,z0+1,d)],   t.x);
        float c11 = mix(s[idx3(x0,y0+1,z0+1,d)], s[idx3(x0+1,y0+1,z0+1,d)], t.x);
        return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
    }

    kernel void scale3d(device float *buf [[buffer(0)]], constant float &k [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) { buf[i] *= k; }

    // viscous diffusion ν∇²v (explicit 6-point 3D Laplacian) — the FAITHFUL fix for
    // fluoddity's uniform drag. Real viscosity is k²-SELECTIVE (kills small scales,
    // spares large ⇒ an inertial range / cascade); uniform drag damped every mode
    // equally ⇒ no cascade. Stable for visc ≤ 1/6 (3D explicit-diffusion limit).
    kernel void diffuse3d(device const float *vin [[buffer(0)]], device float *vout [[buffer(1)]],
                          constant uint3 &d [[buffer(2)]], constant float &visc [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        int x = int(gid.x), y = int(gid.y), z = int(gid.z);
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 c = float3(vin[3*i], vin[3*i+1], vin[3*i+2]);
        float3 lap = vat(vin, x-1,y,z, d) + vat(vin, x+1,y,z, d)
                   + vat(vin, x,y-1,z, d) + vat(vin, x,y+1,z, d)
                   + vat(vin, x,y,z-1, d) + vat(vin, x,y,z+1, d) - 6.0 * c;
        float3 o = c + visc * lap;
        vout[3*i] = o.x; vout[3*i+1] = o.y; vout[3*i+2] = o.z;
    }

    // 1. advect velocity (semi-Lagrangian backtrace)
    kernel void advect3d(device const float *vin [[buffer(0)]], device float *vout [[buffer(1)]],
                         constant uint3 &d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 here = float3(vin[3 * i], vin[3 * i + 1], vin[3 * i + 2]);
        float3 sv = sampleVel3(vin, float3(gid) - here, d);   // cell i lives AT index i (sampler/splat convention) — a +0.5 here advects the whole field by half a cell per frame
        vout[3 * i] = sv.x; vout[3 * i + 1] = sv.y; vout[3 * i + 2] = sv.z;
    }

    // trilinear force deposit at a grid point (cells) — used twice for the dipole
    static inline void depositForce3(device atomic_float *vel, float3 cell, float3 f, uint3 d) {
        float3 fl = floor(cell); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z);
        float3 t = cell - fl;
        for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int ii = 0; ii < 2; ii++) {
            int xi = wrp(x0 + ii, int(d.x)), yi = wrp(y0 + j, int(d.y)), zi = wrp(z0 + k, int(d.z));
            float w = (ii == 0 ? 1 - t.x : t.x) * (j == 0 ? 1 - t.y : t.y) * (k == 0 ? 1 - t.z : t.z);
            uint c = (uint(zi) * d.y + uint(yi)) * d.x + uint(xi);
            atomic_fetch_add_explicit(&vel[3 * c],     f.x * w, memory_order_relaxed);
            atomic_fetch_add_explicit(&vel[3 * c + 1], f.y * w, memory_order_relaxed);
            atomic_fetch_add_explicit(&vel[3 * c + 2], f.z * w, memory_order_relaxed);
        }
    }

    // 2. agents force the fluid as a NET-ZERO force DIPOLE (+f ahead, −f behind ⇒
    //    Σmomentum=0, the faithful active-swimmer forcing — the fix for the monopole
    //    splat) + deposit dye (= local density, the chemotaxis signal) at the center.
    kernel void splat3d(device atomic_float *vel  [[buffer(0)]],
                        device atomic_float *dye   [[buffer(1)]],
                        device const Particle3D *parts [[buffer(2)]],
                        constant uint3 &d          [[buffer(3)]],
                        constant float &gain       [[buffer(4)]],
                        constant float &dyeAmt     [[buffer(5)]],
                        constant float &dipoleLen  [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
        Particle3D p = parts[gid];
        float3 center = p.pos * float3(d) - 0.5;
        float speed = length(p.vel);
        float3 fwd = speed > 1e-6 ? p.vel / speed : float3(1, 0, 0);
        float3 f = p.vel * gain;
        float3 h = fwd * (dipoleLen * 0.5);
        depositForce3(vel, center + h,  f, d);   // +f ahead (head)
        depositForce3(vel, center - h, -f, d);   // −f behind (tail) ⇒ net-zero
        float3 fl = floor(center); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z);
        float3 t = center - fl;
        for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int ii = 0; ii < 2; ii++) {
            int xi = wrp(x0 + ii, int(d.x)), yi = wrp(y0 + j, int(d.y)), zi = wrp(z0 + k, int(d.z));
            float w = (ii == 0 ? 1 - t.x : t.x) * (j == 0 ? 1 - t.y : t.y) * (k == 0 ? 1 - t.z : t.z);
            atomic_fetch_add_explicit(&dye[(uint(zi) * d.y + uint(yi)) * d.x + uint(xi)], dyeAmt * w, memory_order_relaxed);
        }
    }

    // dye advection by the projected field (+ decay)
    kernel void advectDye3d(device const float *din [[buffer(0)]], device float *dout [[buffer(1)]],
                            device const float *v [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                            constant float &decay [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 vel = float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        dout[i] = sampleScalar3(din, float3(gid) - vel, d) * decay;
    }

    // MacCormack dye advection, passes 2+3 (Selle et al. 2008). Plain semi-
    // Lagrangian + trilinear (pass 1 above) is first-order DIFFUSIVE — it blurs
    // the dye a little every frame, smearing creatures into soft blobs. Fix:
    // advect forward (pass 1, decay=1), advect the result BACKWARD (pass 2: same
    // kernel form, +vel), and the round-trip error vs the original estimates the
    // scheme's local truncation error; subtracting half of it (pass 3) cancels
    // the leading diffusive term ⇒ second-order, visibly sharper structures.
    kernel void advectDyeBack3d(device const float *din [[buffer(0)]], device float *dout [[buffer(1)]],
                                device const float *v [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 vel = float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        dout[i] = sampleScalar3(din, float3(gid) + vel, d);
    }

    // pass 3: corrected = fwd + (orig − back)/2, CLAMPED to the min/max of the 8
    // source cells the backtrace interpolated from (the standard limiter — the
    // correction can overshoot and ring; clamping keeps it monotone and the dye
    // non-negative). Decay is applied here, once.
    kernel void maccormackDye3d(device const float *orig [[buffer(0)]],
                                device const float *fwd  [[buffer(1)]],
                                device float *backAndOut [[buffer(2)]],
                                device const float *v    [[buffer(3)]],
                                constant uint3 &d        [[buffer(4)]],
                                constant float &decay    [[buffer(5)]],
                                uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 vel = float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        float corrected = fwd[i] + 0.5 * (orig[i] - backAndOut[i]);
        float3 p = float3(gid) - vel;
        float3 fl = floor(p); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z);
        float lo = INFINITY, hi = -INFINITY;
        for (int dz = 0; dz <= 1; dz++) for (int dy = 0; dy <= 1; dy++) for (int dx = 0; dx <= 1; dx++) {
            float s = orig[idx3(x0 + dx, y0 + dy, z0 + dz, d)];
            lo = min(lo, s); hi = max(hi, s);
        }
        backAndOut[i] = clamp(corrected, lo, hi) * decay;   // own-cell read precedes write — safe
    }

    // 3. divergence
    kernel void divergence3d(device const float *v [[buffer(0)]], device float *divg [[buffer(1)]],
                             constant uint3 &d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        int x = int(gid.x), y = int(gid.y), z = int(gid.z);
        float dx = vat(v, x + 1, y, z, d).x - vat(v, x - 1, y, z, d).x;
        float dy = vat(v, x, y + 1, z, d).y - vat(v, x, y - 1, z, d).y;
        float dz = vat(v, x, y, z + 1, d).z - vat(v, x, y, z - 1, d).z;
        divg[(gid.z * d.y + gid.y) * d.x + gid.x] = 0.5 * (dx + dy + dz);
    }

    // 4. Jacobi iteration of ∇²p = div (6-neighbour)
    kernel void jacobi3d(device const float *pin [[buffer(0)]], device float *pout [[buffer(1)]],
                         device const float *divg [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        int x = int(gid.x), y = int(gid.y), z = int(gid.z);
        float s = pin[idx3(x - 1, y, z, d)] + pin[idx3(x + 1, y, z, d)]
                + pin[idx3(x, y - 1, z, d)] + pin[idx3(x, y + 1, z, d)]
                + pin[idx3(x, y, z - 1, d)] + pin[idx3(x, y, z + 1, d)];
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        pout[i] = (s - divg[i]) / 6.0;
    }

    // 5. subtract pressure gradient → divergence-free
    kernel void subgrad3d(device float *v [[buffer(0)]], device const float *p [[buffer(1)]],
                          constant uint3 &d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        int x = int(gid.x), y = int(gid.y), z = int(gid.z);
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        v[3 * i]     -= 0.5 * (p[idx3(x + 1, y, z, d)] - p[idx3(x - 1, y, z, d)]);
        v[3 * i + 1] -= 0.5 * (p[idx3(x, y + 1, z, d)] - p[idx3(x, y - 1, z, d)]);
        v[3 * i + 2] -= 0.5 * (p[idx3(x, y, z + 1, d)] - p[idx3(x, y, z - 1, d)]);
    }

    // 6. agents: Monte-Carlo tangent-plane Fourier brain → steer in 3D + ride.
    //    Sample random planes that contain vel; run the 2D mirror-averaged brain
    //    on the flow projected into each plane; average the in-plane force into 3D.
    kernel void move3d(device Particle3D *parts   [[buffer(0)]],
                       device const float *v       [[buffer(1)]],
                       constant uint3 &d           [[buffer(2)]],
                       constant MoveParams3 &mp    [[buffer(3)]],
                       device const float4 *rule   [[buffer(4)]],
                       constant uint &frame        [[buffer(5)]],
                       device const float *dye     [[buffer(6)]],
                       constant uint &count        [[buffer(7)]],
                       uint gid [[thread_position_in_grid]]) {
        Particle3D p = parts[gid];
        uint off = ((gid * uint(N_COHORTS3)) / count) * 20;   // this agent's cohort block
        float3 dd = float3(d);
        float speed = length(p.vel);
        float3 fwd = speed > 1e-6 ? p.vel / speed : float3(1.0, 0.0, 0.0);

        // orthonormal basis perpendicular to fwd
        float3 a = abs(fwd.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 perp1 = normalize(cross(fwd, a));
        float3 perp2 = cross(fwd, perp1);

        float ca = cos(mp.sensorAngle), sa = sin(mp.sensorAngle);
        int nS = max(1, int(mp.planeSamples));
        float3 force = float3(0.0);
        for (int s = 0; s < nS; s++) {
            // random in-plane lateral axis w (the plane is span{fwd, w})
            float theta = hashf(float(gid) * 0.0007 + float(s) * 7.13 + float(frame) * 0.0173) * 6.2831853;
            float3 w = perp1 * cos(theta) + perp2 * sin(theta);

            float3 ldir = fwd * ca + w * sa;
            float3 rdir = fwd * ca - w * sa;
            float3 vL = sampleVel3(v, (p.pos + ldir * mp.sensorDist) * dd, d);
            float3 vR = sampleVel3(v, (p.pos + rdir * mp.sensorDist) * dd, d);
            float2 L = float2(dot(vL, fwd), dot(vL, w)) * mp.senseScale;
            float2 R = float2(dot(vR, fwd), dot(vR, w)) * mp.senseScale;

            float4 base = fourier3(rule, off, float4(L.x, L.y, R.x, R.y));
            float4 mir  = fourier3(rule, off, float4(R.x, -R.y, L.x, -L.y));
            float axial = base.x + mir.x;       // reflection-even
            float turnT = base.y - mir.y;       // reflection-odd (no handedness)
            force += fwd * (axial * mp.axialForce) + w * (turnT * mp.turn);
        }
        force /= float(nS);

        // chemotaxis: steer up the dye (density) gradient → aggregation. THIS is the
        // creature-maker (AT-5); without it velocity-sensing agents stay uniform (AT-4).
        float sd = mp.sensorDist;
        float gx = sampleScalar3(dye, (p.pos + float3(sd,0,0)) * dd, d) - sampleScalar3(dye, (p.pos - float3(sd,0,0)) * dd, d);
        float gy = sampleScalar3(dye, (p.pos + float3(0,sd,0)) * dd, d) - sampleScalar3(dye, (p.pos - float3(0,sd,0)) * dd, d);
        float gz = sampleScalar3(dye, (p.pos + float3(0,0,sd)) * dd, d) - sampleScalar3(dye, (p.pos - float3(0,0,sd)) * dd, d);
        force += float3(gx, gy, gz) * mp.cohesion;

        float3 nv = p.vel + force;
        float ns = length(nv);
        nv = ns > 1e-6 ? nv / ns * clamp(ns, mp.swim * 0.3, mp.swim * 2.0) : fwd * mp.swim;
        p.vel = nv;                              // brain-steered velocity = fluid forcing

        float3 fv = sampleVel3(v, p.pos * dd, d);
        p.pos = fract(p.pos + p.vel * AGENT_DT3 + (fv / dd) * mp.fluidPull);
        parts[gid] = p;
    }

    // agent point overlay, COHORT-TINTED (the species identity the shared dye
    // volume can't show — turn up pointAlpha to see who's who when breeding)
    constant float3 COHORT_COLS[8] = {
        float3(1.0, 0.35, 0.35), float3(1.0, 0.75, 0.2), float3(0.95, 1.0, 0.3), float3(0.3, 1.0, 0.45),
        float3(0.3, 0.95, 1.0),  float3(0.35, 0.5, 1.0), float3(0.75, 0.4, 1.0), float3(1.0, 0.4, 0.85)
    };
    struct P3Out { float4 position [[position]]; float pointSize [[point_size]]; float3 col; };
    vertex P3Out point3d_vertex(uint vid                       [[vertex_id]],
                                device const Particle3D *parts  [[buffer(0)]],
                                constant float4x4 &viewProj     [[buffer(1)]],
                                constant uint &count            [[buffer(2)]]) {
        Particle3D p = parts[vid];
        P3Out o;
        o.position = viewProj * float4(p.pos, 1.0);
        o.pointSize = 2.5;
        o.col = COHORT_COLS[(vid * uint(N_COHORTS3)) / count];
        return o;
    }
    fragment float4 point3d_fragment(P3Out in [[stage_in]], float2 pc [[point_coord]],
                                     constant float &alpha [[buffer(0)]]) {
        float r = length(pc - 0.5) * 2.0;
        if (r > 1.0) discard_fragment();
        float a = (1.0 - r * r) * alpha;
        return float4(in.col * a, a);
    }

    // vorticity ω = ∇×u via central differences (cells/frame units) — the
    // fluid-only volume view samples this instead of the agent-deposited dye
    static inline float3 curl3(device const float *v, float3 p, uint3 d) {
        float3 xp = sampleVel3(v, p + float3(1,0,0), d), xm = sampleVel3(v, p - float3(1,0,0), d);
        float3 yp = sampleVel3(v, p + float3(0,1,0), d), ym = sampleVel3(v, p - float3(0,1,0), d);
        float3 zp = sampleVel3(v, p + float3(0,0,1), d), zm = sampleVel3(v, p - float3(0,0,1), d);
        return 0.5 * float3((yp.z - ym.z) - (zp.y - zm.y),
                            (zp.x - zm.x) - (xp.z - xm.z),
                            (xp.y - xm.y) - (yp.x - ym.x));
    }

    // ── volumetric ray-march of the dye field (Stage 3) ───────────────────
    struct RMOut { float4 position [[position]]; float2 uv; };
    vertex RMOut raymarch_vertex(uint vid [[vertex_id]]) {
        float2 uv = float2((vid << 1) & 2, vid & 2);
        RMOut o; o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0); o.uv = uv; return o;
    }
    fragment float4 raymarch_fragment(RMOut in                    [[stage_in]],
                                      device const float *dye      [[buffer(0)]],
                                      constant uint3 &d            [[buffer(1)]],
                                      constant float4x4 &invVP     [[buffer(2)]],
                                      constant float &densityScale [[buffer(3)]],
                                      device const float *vfield   [[buffer(4)]],
                                      constant float &colorMode    [[buffer(5)]],
                                      constant float &sharpness    [[buffer(6)]],
                                      constant float &vortScale    [[buffer(7)]]) {
        int cm = int(colorMode + 0.5);
        float3 bg = float3(0.02, 0.02, 0.05);
        float2 ndc = in.uv * 2.0 - 1.0;
        float4 nh = invVP * float4(ndc, 0.0, 1.0);
        float4 fh = invVP * float4(ndc, 1.0, 1.0);
        float3 ro = nh.xyz / nh.w;
        float3 rd = normalize(fh.xyz / fh.w - ro);
        float3 invd = 1.0 / rd;
        float3 t0 = (float3(0.0) - ro) * invd, t1 = (float3(1.0) - ro) * invd;
        float3 tmn = min(t0, t1), tmx = max(t0, t1);
        float tn = max(max(tmn.x, tmn.y), tmn.z);
        float tf = min(min(tmx.x, tmx.y), tmx.z);
        if (tf <= max(tn, 0.0)) return float4(bg, 1.0);
        tn = max(tn, 0.0);
        const int STEPS = 224;   // ~1 sample/cell across a 160³ volume (was 96 → undersampled/banded)
        float dt = (tf - tn) / float(STEPS);
        float3 acc = float3(0.0); float trans = 1.0;
        for (int i = 0; i < STEPS; i++) {
            float3 pos = ro + (tn + (float(i) + 0.5) * dt) * rd;
            // sharpness = transfer-function gamma: >1 crushes the faint trilinear
            // halo around structures and keeps the cores ⇒ crisper boundaries on
            // zoom (the data is 160³ voxels; this sharpens the MAPPING, not the data)
            float dens;
            float3 vortEmit = float3(0.0);
            if (cm == 3) {
                // fluid-only view: opacity from |∇×u| (vortex tubes — where the
                // dynamics live), color from the vorticity DIRECTION. The agents'
                // dye plays no part; you see the NS fluid itself.
                float3 om = curl3(vfield, pos * float3(d), d);
                float w = length(om);
                dens = pow(w * vortScale, sharpness);
                vortEmit = 0.5 + 0.5 * (w > 1e-6 ? om / w : float3(0.0));
            } else {
                dens = pow(max(sampleScalar3(dye, pos * float3(d), d), 0.0) * densityScale, sharpness);
            }
            if (dens > 0.002) {
                float a = 1.0 - exp(-dens * dt * 15.0);
                float3 emit;
                if (cm == 3) {                        // vorticity direction → RGB
                    emit = vortEmit;
                } else if (cm == 0) {                 // density gradient
                    emit = mix(float3(0.1, 0.4, 0.9), float3(1.0, 0.6, 0.2), clamp(dens, 0.0, 1.0));
                } else {
                    float3 fv = sampleVel3(vfield, pos * float3(d), d);
                    if (cm == 1) emit = 0.5 + 0.5 * (length(fv) > 1e-6 ? normalize(fv) : float3(0.0));  // flow direction → RGB
                    else         emit = mix(float3(0.1, 0.3, 0.8), float3(1.0, 0.9, 0.4), clamp(length(fv) * 4.0, 0.0, 1.0));  // speed
                }
                acc += trans * a * emit;
                trans *= 1.0 - a;
                if (trans < 0.01) break;
            }
        }
        return float4(acc + bg * trans, 1.0);
    }
    """
}
