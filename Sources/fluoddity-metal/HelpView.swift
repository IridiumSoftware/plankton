import AppKit

// A clickable container that calls back on mouseDown (used to collapse the help).
final class HelpClickView: NSView {
    var onClick: (() -> Void)?
    override func mouseDown(with e: NSEvent) { onClick?() }
}

// A small "⌨ Controls" button that expands to a translucent panel listing the
// keyboard/mouse controls; clicking the panel collapses it back to the button.
// Anchor it bottom-left — it grows upward when expanded.
final class HelpView: NSView {
    private let button = NSButton(title: "⌨ Controls", target: nil, action: nil)
    private let panel = HelpClickView()
    private let collapsedSize: NSSize
    private let expandedSize: NSSize

    init(text: String) {
        let lines = text.components(separatedBy: "\n")
        let w: CGFloat = 300
        let h = CGFloat(lines.count) * 16 + 18
        collapsedSize = NSSize(width: 110, height: 26)
        expandedSize = NSSize(width: w, height: h)
        super.init(frame: NSRect(origin: .zero, size: collapsedSize))

        button.bezelStyle = .rounded
        button.font = .systemFont(ofSize: 11)
        button.frame = NSRect(origin: .zero, size: collapsedSize)
        button.target = self
        button.action = #selector(expand)
        addSubview(button)

        panel.frame = NSRect(origin: .zero, size: expandedSize)
        panel.wantsLayer = true
        panel.layer?.backgroundColor = NSColor(white: 0, alpha: 0.8).cgColor
        panel.layer?.cornerRadius = 8
        panel.isHidden = true
        panel.onClick = { [weak self] in self?.collapse() }

        let lbl = NSTextField(frame: NSRect(x: 12, y: 8, width: w - 24, height: h - 16))
        lbl.stringValue = text
        lbl.isEditable = false; lbl.isBordered = false; lbl.drawsBackground = false; lbl.isSelectable = false
        lbl.textColor = .white
        lbl.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        lbl.usesSingleLineMode = false
        lbl.maximumNumberOfLines = 0
        lbl.lineBreakMode = .byWordWrapping
        panel.addSubview(lbl)
        addSubview(panel)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    @objc private func expand() {
        setFrameSize(expandedSize)   // grows upward (bottom-left anchored)
        button.isHidden = true
        panel.isHidden = false
    }
    private func collapse() {
        setFrameSize(collapsedSize)
        panel.isHidden = true
        button.isHidden = false
    }
}
