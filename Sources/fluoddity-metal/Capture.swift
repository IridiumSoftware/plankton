import Foundation

// Capture & replay — two ways to keep a creature you grew (the system is hysteretic,
// AT-7: a creature is a function of its history, not the parameter values):
//   • CREATURE  — the full state (agents + fields + brains + params) → one `.fluo` file.
//                 Restores the exact configuration, bit-for-bit.
//   • PATH      — the developmental trajectory: the start state + the timeline of every
//                 parameter change → a `.fluopath` file. Replaying re-grows the creature
//                 along the same path (the way you actually reached it).
// Files live under ./captures/{creatures,paths} relative to the run directory.
enum Captures {
    private static func dir(_ sub: String) -> URL {
        let d = URL(fileURLWithPath: "captures/\(sub)", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    static func list(_ sub: String, _ ext: String) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir(sub), includingPropertiesForKeys: nil)) ?? []
        return items.filter { $0.pathExtension == ext }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
    static func nextURL(_ sub: String, _ prefix: String, _ ext: String) -> URL {
        let n = list(sub, ext).count + 1
        return dir(sub).appendingPathComponent(String(format: "%@_%03d.%@", prefix, n, ext))
    }
}

// Records the parameter trajectory (start state + per-frame param deltas) and replays it.
// The render loop calls tickRecord/tickReplay each frame; only one mode is active at a time.
final class PathJournal {
    private(set) var recording = false
    private(set) var replaying = false
    private var frame = 0
    private var last: [Float] = []
    private var changes: [(Int32, Int32, Float)] = []     // (frame, knobIndex, value)
    private var startState = Data()

    // begin recording from the current canvas (pass the full serialized start state)
    func startRecording(startState: Data, _ params: Params) {
        self.startState = startState; changes.removeAll(); frame = 0
        last = engineKnobs.map { params[keyPath: $0.kp] }
        recording = true; replaying = false
        print("● recording path — tune to grow a creature, press j again to stop")
    }

    // call once per frame while recording: snapshot params, record any that changed
    func tickRecord(_ params: Params) {
        guard recording else { return }
        frame += 1
        for (i, k) in engineKnobs.enumerated() {
            let v = params[keyPath: k.kp]
            if v != last[i] { changes.append((Int32(frame), Int32(i), v)); last[i] = v }
        }
    }

    @discardableResult
    func stopRecordingAndSave() -> URL? {
        recording = false
        var data = Data()
        func putI(_ v: Int32) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        putI(Int32(startState.count)); data.append(startState)
        putI(Int32(changes.count))
        for (f, i, v) in changes { putI(f); putI(i); var vv = v; withUnsafeBytes(of: &vv) { data.append(contentsOf: $0) } }
        let url = Captures.nextURL("paths", "path", "fluopath")
        try? data.write(to: url)
        print("■ saved \(url.lastPathComponent) — \(changes.count) param changes over \(frame) frames")
        return url
    }

    // load a path: returns the start state for the caller to apply, then per-frame tickReplay
    func beginReplay(_ url: URL) -> Data? {
        guard let data = try? Data(contentsOf: url) else { print("path: cannot read \(url.lastPathComponent)"); return nil }
        var start: Data? = nil
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard raw.count >= 8 else { return }
            var off = 0
            func i32() -> Int32 { let v = raw.loadUnaligned(fromByteOffset: off, as: Int32.self); off += 4; return v }
            let slen = Int(i32())
            guard slen > 0, off + slen + 4 <= raw.count else { return }
            start = data.subdata(in: off ..< off + slen); off += slen
            let nc = Int(i32()); changes.removeAll()
            for _ in 0..<nc {
                let f = i32(); let i = i32()
                let v = raw.loadUnaligned(fromByteOffset: off, as: Float.self); off += 4
                changes.append((f, i, v))
            }
        }
        guard start != nil else { print("path: malformed \(url.lastPathComponent)"); return nil }
        frame = 0; replaying = true; recording = false
        print("▶ replaying \(url.lastPathComponent) — \(changes.count) changes")
        return start
    }

    // call once per frame while replaying: apply any param changes scheduled for this frame
    func tickReplay(_ params: Params) {
        guard replaying else { return }
        for (f, i, v) in changes where Int(f) == frame { params[keyPath: engineKnobs[Int(i)].kp] = v }
        frame += 1
        if frame > Int(changes.last?.0 ?? 0) + 180 { replaying = false; print("▶ replay done") }
    }
}
