import AppKit
import Foundation
import Vision

let arguments = CommandLine.arguments
if arguments.count < 3 {
    fputs("usage: vision-ocr ocr FILE | render FILE TEXT\n", stderr)
    exit(64)
}

if arguments[1] == "render" {
    let text = arguments.count > 3 ? arguments[3] : ""
    let size = NSSize(width: 900, height: 220)
    let image = NSImage(size: size)
    image.lockFocus()
    NSColor.white.setFill()
    NSRect(origin: .zero, size: size).fill()
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 32),
        .foregroundColor: NSColor.black,
    ]
    (text as NSString).draw(at: NSPoint(x: 24, y: 90), withAttributes: attributes)
    image.unlockFocus()
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else {
        exit(2)
    }
    do {
        try png.write(to: URL(fileURLWithPath: arguments[2]))
    } catch {
        fputs("\(error)\n", stderr)
        exit(2)
    }
    exit(0)
}

let url = URL(fileURLWithPath: arguments[2])
guard let image = NSImage(contentsOf: url),
      let tiff = image.tiffRepresentation,
      let ciImage = CIImage(data: tiff) else {
    fputs("unreadable\n", stderr)
    exit(2)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("\(error)\n", stderr)
    exit(3)
}
for observation in request.results ?? [] {
    if let line = observation.topCandidates(1).first?.string {
        print(line)
    }
}
