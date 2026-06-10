import simd

// Orbital perspective camera around the unit-cube center. View3D drives
// azimuth / elevation / distance from mouse input; Renderer3D reads viewProj().
final class Camera3D {
    var azimuth: Float = 0.6
    var elevation: Float = 0.4
    var distance: Float = 2.2
    var aspect: Float = 1.0
    let center = SIMD3<Float>(0.5, 0.5, 0.5)

    func viewProj() -> float4x4 {
        let ce = cos(elevation), se = sin(elevation)
        let ca = cos(azimuth), sa = sin(azimuth)
        let eye = center + distance * SIMD3(ce * sa, se, ce * ca)
        let view = Camera3D.lookAt(eye, center, SIMD3(0, 1, 0))
        let proj = Camera3D.perspective(45 * .pi / 180, aspect, 0.01, 10)
        return proj * view
    }

    func invViewProj() -> float4x4 { viewProj().inverse }

    static func perspective(_ fovy: Float, _ aspect: Float, _ near: Float, _ far: Float) -> float4x4 {
        let ys = 1 / tan(fovy * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        return float4x4(columns: (
            SIMD4(xs, 0, 0, 0),
            SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, zs, -1),
            SIMD4(0, 0, zs * near, 0)))
    }

    static func lookAt(_ eye: SIMD3<Float>, _ c: SIMD3<Float>, _ up: SIMD3<Float>) -> float4x4 {
        let f = normalize(c - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)
        return float4x4(columns: (
            SIMD4(s.x, u.x, -f.x, 0),
            SIMD4(s.y, u.y, -f.y, 0),
            SIMD4(s.z, u.z, -f.z, 0),
            SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)))
    }
}
