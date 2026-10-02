import UIKit

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions _: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        // A real window keeps the app in the foreground — iOS forbids
        // GPU work in the background, so the run must stay foregrounded.
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = PlanesViewController()
        window.makeKeyAndVisible()
        self.window = window

        application.isIdleTimerDisabled = true
        // The brightness is owned by the Rust side: it dims only after
        // the launch arguments validate, and restores on resign-active,
        // terminate and every fatal exit path.
        return true
    }

    func applicationWillResignActive(_ application: UIApplication) {
        cherenkov_planes_brightness_restore()
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        cherenkov_planes_brightness_dim()
    }

    func applicationWillTerminate(_ application: UIApplication) {
        cherenkov_planes_brightness_restore()
    }
}
