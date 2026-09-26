// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import Foundation
import OSLog

private let logger = Logger(subsystem: "dev.cherenkov", category: "bench")

/// Runs every argument list in `Documents/bench-args.json` through
/// `cherenkov_bench_run`, in order, and records each run's exit code in
/// `Documents/out/done.json`.
struct BenchRunner {
    /// The overall exit code: the first non-zero run's, else 0.
    func runAll() -> Int32 {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let outDir = documents.appendingPathComponent("out", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        } catch {
            logger.error("cannot create \(outDir.path, privacy: .public): \(error)")
            return 1
        }
        // An iOS app launches with cwd `/`; the bench's relative
        // `Documents/...` paths only resolve from the app's home.
        guard FileManager.default.changeCurrentDirectoryPath(NSHomeDirectory()) else {
            logger.error("cannot chdir to \(NSHomeDirectory(), privacy: .public)")
            _ = Self.writeDone(["error": "cannot chdir to app home"], to: outDir)
            return 1
        }
        guard let argLists = Self.loadArgLists(from: documents.appendingPathComponent("bench-args.json"))
        else {
            _ = Self.writeDone(["error": "bench-args.json missing or invalid"], to: outDir)
            return 1
        }
        var results: [[String: Any]] = []
        var firstFailure: Int32 = 0
        for (index, args) in argLists.enumerated() {
            logger.info("run \(index): \(args.joined(separator: " "), privacy: .public)")
            let stderrURL = outDir.appendingPathComponent("run-\(index).stderr")
            let code = Self.invoke(args, stderrTo: stderrURL)
            logger.info("run \(index): exit \(code)")
            results.append(["args": args, "exit_code": code])
            if firstFailure == 0 { firstFailure = code }
        }
        guard Self.writeDone(["results": results], to: outDir) else { return 1 }
        return firstFailure
    }

    /// Calls `cherenkov_bench_run` with `cherenkov-bench` as `argv[0]`
    /// followed by `args`, with fd 2 redirected to `stderrURL` for the
    /// duration of the call and restored afterwards.
    static func invoke(_ args: [String], stderrTo stderrURL: URL) -> Int32 {
        var cArgs = (["cherenkov-bench"] + args).map { strdup($0) }
        defer { cArgs.forEach { free($0) } }

        let saved = dup(STDERR_FILENO)
        let log = open(stderrURL.path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        let redirected = saved >= 0 && log >= 0
        if redirected {
            dup2(log, STDERR_FILENO)
        } else {
            logger.error("stderr redirect to \(stderrURL.path, privacy: .public) failed (saved=\(saved), log=\(log))")
        }
        if log >= 0 { close(log) }

        let code = cArgs.withUnsafeMutableBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(
                to: UnsafePointer<CChar>?.self,
                capacity: buffer.count
            ) { argv in
                cherenkov_bench_run(Int32(buffer.count), argv)
            }
        }

        fflush(nil)
        if redirected { dup2(saved, STDERR_FILENO) }
        if saved >= 0 { close(saved) }
        return code
    }

    static func loadArgLists(from url: URL) -> [[String]]? {
        guard let data = try? Data(contentsOf: url),
              let lists = try? JSONSerialization.jsonObject(with: data) as? [[String]]
        else {
            logger.error("\(url.lastPathComponent, privacy: .public) missing or not an array of argument lists")
            return nil
        }
        return lists
    }

    @discardableResult
    static func writeDone(_ object: [String: Any], to outDir: URL) -> Bool {
        let url = outDir.appendingPathComponent("done.json")
        do {
            let data = try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: url, options: .atomic)
            logger.info("wrote \(url.path, privacy: .public)")
            return true
        } catch {
            logger.error("cannot write \(url.path, privacy: .public): \(error)")
            return false
        }
    }
}
