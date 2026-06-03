// 3D engine shaders (Stage 1): particles advected by an analytic ABC flow
// (Arnold–Beltrami–Childress — a divergence-free 3D flow with chaotic
// streamlines), rendered as an additive point cloud through an orbital camera.
// Stage 2 replaces the analytic flow with a real 3D fluid grid + agent physics.
enum Shaders3D {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Particle3D { float3 pos; float3 vel; };

    constant float ABC_A = 1.0, ABC_B = 0.7, ABC_C = 0.43;
    constant float FLOW_SPEED = 0.06;
    constant float DT3 = 0.016;

    static inline float3 abcFlow(float3 p) {
        float x = p.x * 6.2831853, y = p.y * 6.2831853, z = p.z * 6.2831853;
        return float3(ABC_A * sin(z) + ABC_C * cos(y),
                      ABC_B * sin(x) + ABC_A * cos(z),
                      ABC_C * sin(y) + ABC_B * cos(x));
    }

    // advect each particle along the ABC flow; wrap in the unit cube (periodic)
    kernel void move3d(device Particle3D *parts [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
        Particle3D p = parts[gid];
        float3 v = abcFlow(p.pos) * FLOW_SPEED;
        p.vel = v;
        p.pos = fract(p.pos + v * DT3);
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
        float t = clamp(in.speed * 8.0, 0.0, 1.0);     // slow → blue, fast → warm
        float3 col = mix(float3(0.2, 0.5, 1.0), float3(1.0, 0.7, 0.3), t);
        return float4(col * a, a);
    }
    """
}
