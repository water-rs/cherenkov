import OSLog
import UIKit

private let logger = Logger(subsystem: "dev.cherenkov", category: "bench")

/// Owns the one window. iOS 27 stops a process that reaches
/// `UIApplicationMain` with no scene manifest: the crash is
/// `EXC_BREAKPOINT` in
/// `UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption`,
/// signal 5, before the bench thread can write `thermal.json`.
@objc(SceneDelegate)
final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    private var originalBrightness: CGFloat = 0
    private static var started = false

    func scene(
        _ scene: UIScene,
        willConnectTo _: UISceneSession,
        options _: UIScene.ConnectionOptions
    ) {
        guard let windowScene = scene as? UIWindowScene else { return }
        let window = UIWindow(windowScene: windowScene)
        let status = StatusViewController()
        window.rootViewController = status
        window.makeKeyAndVisible()
        self.window = window

        // A scene reconnect must not start a second bench over the first.
        guard !Self.started else { return }
        Self.started = true

        // A real window keeps the app in the foreground — iOS forbids
        // GPU work in the background, so the run must stay foregrounded.
        let application = UIApplication.shared
        application.isIdleTimerDisabled = true
        originalBrightness = windowScene.screen.brightness
        windowScene.screen.brightness = 0
        logger.info("launch: idle timer disabled, brightness -> 0")

        let screen = windowScene.screen
        Thread.detachNewThread { [weak self] in
            let code = BenchRunner(delegate: status).runAll()
            let brightness = self?.originalBrightness
            DispatchQueue.main.async {
                application.isIdleTimerDisabled = false
                if let brightness {
                    screen.brightness = brightness
                }
            }
            logger.info("all runs done; exit code \(code)")
        }
    }
}
