import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def test_pages_and_demo_processing(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        for page in [
            "Frame Analysis",
            "Performance Dashboard",
            "Deep Vision",
            "Export Center",
            "About Project",
            "Live Vision Pipeline",
        ]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)
        app.button[0].click().run(timeout=90)
        self.assertFalse(app.exception)
        self.assertIn("runDir", app.session_state)
        for page in ["Frame Analysis", "Performance Dashboard", "Export Center"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)


if __name__ == "__main__":
    unittest.main()
