import Foundation

// A saved preset = every slider value (by knob name) + the 80-param brain.
// Stored as JSON under ./presets (relative to the run directory).
struct PresetData: Codable {
    var params: [String: Float]
    var rule: [Float]
}

enum Presets {
    static func dir() -> URL {
        let d = URL(fileURLWithPath: "presets", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func list() -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: dir(), includingPropertiesForKeys: nil)) ?? []
        return items.filter { $0.pathExtension == "json" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    @discardableResult
    static func save(_ data: PresetData) -> URL? {
        let n = list().count + 1
        let url = dir().appendingPathComponent(String(format: "preset_%03d.json", n))
        guard let json = try? JSONEncoder().encode(data) else { return nil }
        try? json.write(to: url)
        return url
    }

    static func load(_ url: URL) -> PresetData? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(PresetData.self, from: data)
    }
}
