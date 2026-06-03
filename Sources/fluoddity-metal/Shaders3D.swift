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
    struct MoveParams3 { float swim; float sensorDist; float sensorAngle; float turn; float axialForce; float fluidPull; float senseScale; float planeSamples; };
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
    static inline float4 fourier3(device const float4 *rule, float4 x) {
        float4 acc = float4(0.0);
        for (int i = 0; i < 10; i++) acc += rule[10 + i] * sin(dot(rule[i], x));
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

    // 1. advect velocity (semi-Lagrangian backtrace)
    kernel void advect3d(device const float *vin [[buffer(0)]], device float *vout [[buffer(1)]],
                         constant uint3 &d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 here = float3(vin[3 * i], vin[3 * i + 1], vin[3 * i + 2]);
        float3 sv = sampleVel3(vin, float3(gid) + 0.5 - here, d);
        vout[3 * i] = sv.x; vout[3 * i + 1] = sv.y; vout[3 * i + 2] = sv.z;
    }

    // 2. splat agent velocities (forcing) + dye (volume density), trilinear atomic
    kernel void splat3d(device atomic_float *vel  [[buffer(0)]],
                        device atomic_float *dye   [[buffer(1)]],
                        device const Particle3D *parts [[buffer(2)]],
                        constant uint3 &d          [[buffer(3)]],
                        constant float &gain       [[buffer(4)]],
                        constant float &dyeAmt     [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
        Particle3D p = parts[gid];
        float3 cell = p.pos * float3(d) - 0.5;
        float3 fl = floor(cell); int x0 = int(fl.x), y0 = int(fl.y), z0 = int(fl.z);
        float3 t = cell - fl;
        float3 f = p.vel * gain;
        for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int ii = 0; ii < 2; ii++) {
            int xi = wrp(x0 + ii, int(d.x)), yi = wrp(y0 + j, int(d.y)), zi = wrp(z0 + k, int(d.z));
            float w = (ii == 0 ? 1 - t.x : t.x) * (j == 0 ? 1 - t.y : t.y) * (k == 0 ? 1 - t.z : t.z);
            uint c = (uint(zi) * d.y + uint(yi)) * d.x + uint(xi);
            atomic_fetch_add_explicit(&vel[3 * c],     f.x * w, memory_order_relaxed);
            atomic_fetch_add_explicit(&vel[3 * c + 1], f.y * w, memory_order_relaxed);
            atomic_fetch_add_explicit(&vel[3 * c + 2], f.z * w, memory_order_relaxed);
            atomic_fetch_add_explicit(&dye[c],         dyeAmt * w, memory_order_relaxed);
        }
    }

    // dye advection by the projected field (+ decay)
    kernel void advectDye3d(device const float *din [[buffer(0)]], device float *dout [[buffer(1)]],
                            device const float *v [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                            constant float &decay [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.x || gid.y >= d.y || gid.z >= d.z) return;
        uint i = (gid.z * d.y + gid.y) * d.x + gid.x;
        float3 vel = float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        dout[i] = sampleScalar3(din, float3(gid) + 0.5 - vel, d) * decay;
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
                       uint gid [[thread_position_in_grid]]) {
        Particle3D p = parts[gid];
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

            float4 base = fourier3(rule, float4(L.x, L.y, R.x, R.y));
            float4 mir  = fourier3(rule, float4(R.x, -R.y, L.x, -L.y));
            float axial = base.x + mir.x;       // reflection-even
            float turnT = base.y - mir.y;       // reflection-odd (no handedness)
            force += fwd * (axial * mp.axialForce) + w * (turnT * mp.turn);
        }
        force /= float(nS);

        float3 nv = p.vel + force;
        float ns = length(nv);
        nv = ns > 1e-6 ? nv / ns * clamp(ns, mp.swim * 0.3, mp.swim * 2.0) : fwd * mp.swim;
        p.vel = nv;                              // brain-steered velocity = fluid forcing

        float3 fv = sampleVel3(v, p.pos * dd, d);
        p.pos = fract(p.pos + p.vel * AGENT_DT3 + (fv / dd) * mp.fluidPull);
        parts[gid] = p;
    }

    struct P3Out { float4 position [[position]]; float pointSize [[point_size]]; float speed; };
    vertex P3Out point3d_vertex(uint vid                       [[vertex_id]],
                                device const Particle3D *parts  [[buffer(0)]],
                                constant float4x4 &viewProj     [[buffer(1)]]) {
        Particle3D p = parts[vid];
        P3Out o;
        o.position = viewProj * float4(p.pos, 1.0);
        o.pointSize = 2.0;
        o.speed = length(p.vel);
        return o;
    }
    fragment float4 point3d_fragment(P3Out in [[stage_in]], float2 pc [[point_coord]]) {
        float r = length(pc - 0.5) * 2.0;
        if (r > 1.0) discard_fragment();
        float a = (1.0 - r * r) * 0.12;
        float t = clamp(in.speed * 4.0, 0.0, 1.0);
        float3 col = mix(float3(0.2, 0.5, 1.0), float3(1.0, 0.7, 0.3), t);
        return float4(col * a, a);
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
                                      constant float &colorMode    [[buffer(5)]]) {
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
        const int STEPS = 96;
        float dt = (tf - tn) / float(STEPS);
        float3 acc = float3(0.0); float trans = 1.0;
        for (int i = 0; i < STEPS; i++) {
            float3 pos = ro + (tn + (float(i) + 0.5) * dt) * rd;
            float dens = sampleScalar3(dye, pos * float3(d), d) * densityScale;
            if (dens > 0.002) {
                float a = 1.0 - exp(-dens * dt * 15.0);
                float3 emit;
                if (cm == 0) {                        // density gradient
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
