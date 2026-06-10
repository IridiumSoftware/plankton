import Foundation

// Shared output location for the headless study modes (--sweep / --map / --map3 /
// --sdscan / --bistab / --3dspec). They all write their CSVs (and the 3dspec
// velocity-field .bin dumps) under data/ to keep the repo root clean; the Python
// analysis scripts read from the same place. Paths are relative to the cwd, which
// is the repo root (where presets/ and figures/ also live).
enum Study {
    static let dir = "data"

    static func ensureDir() {
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    /// data/<name> as a string (for embedding in a manifest, e.g. the bin paths).
    static func path(_ name: String) -> String { "\(dir)/\(name)" }

    /// data/<name> as a URL, ensuring data/ exists first.
    static func url(_ name: String) -> URL { ensureDir(); return URL(fileURLWithPath: path(name)) }
}
