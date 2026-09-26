// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

import OSLog
import UIKit

private let logger = Logger(subsystem: "dev.cherenkov", category: "bench")

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    private var originalBrightness: CGFloat = UIScreen.main.brightness

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions _: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        // A real window keeps the app in the foreground — iOS forbids
        // GPU work in the background, so the run must stay foregrounded.
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = UIViewController()
        window.makeKeyAndVisible()
        self.window = window

        application.isIdleTimerDisabled = true
        originalBrightness = UIScreen.main.brightness
        UIScreen.main.brightness = 0
        logger.info("launch: idle timer disabled, brightness -> 0")

        Thread.detachNewThread { [weak self] in
            let code = BenchRunner().runAll()
            DispatchQueue.main.sync {
                application.isIdleTimerDisabled = false
                if let brightness = self?.originalBrightness {
                    UIScreen.main.brightness = brightness
                }
            }
            logger.info("all runs done; exit \(code)")
            exit(code)
        }
        return true
    }
}
