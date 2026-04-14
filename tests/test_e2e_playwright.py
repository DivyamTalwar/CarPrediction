"""
End-to-end Playwright test - simulates the exact demo flow.
Tests the full UI: load page -> fill form -> submit -> verify prediction.
"""
import subprocess
import time
import sys
import os

import pytest
import requests
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PORT = "8092"
BASE_URL = os.environ.get("BASE_URL", f"http://127.0.0.1:{DEFAULT_PORT}")
SCREENSHOTS_DIR = os.path.join(PROJECT_DIR, "outputs", "screenshots")


def setup_screenshots_dir():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def wait_for_server(base_url, timeout=20):
    """Wait until the Flask app responds on /health."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError(f"Server did not become ready at {base_url}")


@pytest.fixture(scope="module")
def live_server():
    """Start the local Flask app unless an external BASE_URL is provided."""
    if "BASE_URL" in os.environ:
        wait_for_server(BASE_URL)
        yield BASE_URL
        return

    env = os.environ.copy()
    env["PORT"] = DEFAULT_PORT
    process = subprocess.Popen(
        [sys.executable, "api/app.py"],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_server(BASE_URL)
        yield BASE_URL
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_full_demo_flow(live_server):
    """Simulate the exact demo: open page, fill form, predict, verify."""
    setup_screenshots_dir()

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError as exc:
            pytest.skip(f"Playwright browser not available: {exc}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # STEP 1: Load the home page
        print("\n[DEMO STEP 1] Loading home page...")
        page.goto(live_server)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)  # Wait for stagger animations

        # Verify page loaded
        assert page.title() == "Car Price Prediction"
        heading = page.locator("h1").inner_text()
        assert "Car Price" in heading
        assert "Prediction" in heading
        print(f"  Page loaded. Title OK")

        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "01_home_page.png"), full_page=True)
        print("  Screenshot: 01_home_page.png")

        # STEP 2: Fill the form - Test Case 1: Maruti Swift
        print("\n[DEMO STEP 2] Test Case 1: Maruti Swift, Petrol, Manual, 3 years old")
        page.select_option("#brand", "Maruti")
        page.fill("#present_price", "7.5")
        page.fill("#kms_driven", "25000")
        page.select_option("#fuel_type", "Petrol")
        page.select_option("#seller_type", "Dealer")
        page.select_option("#transmission", "Manual")
        page.select_option("#owner", "0")
        page.fill("#car_age", "3")

        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "02_form_filled_maruti.png"), full_page=True)
        print("  Screenshot: 02_form_filled_maruti.png")

        # Submit via button (force=True to bypass animation opacity)
        page.click("#predictBtn", force=True)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        # Verify prediction appears
        result = page.locator(".result")
        assert result.is_visible(), "Result section should be visible"
        price_text = page.locator(".result__price").inner_text()
        assert "Lakhs" in price_text, f"Expected 'Lakhs' in price, got: {price_text}"
        print(f"  PREDICTED: {price_text}")

        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "03_prediction_maruti.png"), full_page=True)
        print("  Screenshot: 03_prediction_maruti.png")

        # STEP 3: Test Case 2: BMW Automatic
        print("\n[DEMO STEP 3] Test Case 2: BMW, Diesel, Automatic, 2 years old")
        page.select_option("#brand", "BMW")
        page.fill("#present_price", "45.0")
        page.fill("#kms_driven", "15000")
        page.select_option("#fuel_type", "Diesel")
        page.select_option("#seller_type", "Dealer")
        page.select_option("#transmission", "Automatic")
        page.select_option("#owner", "0")
        page.fill("#car_age", "2")

        page.click("#predictBtn", force=True)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        result = page.locator(".result")
        assert result.is_visible()
        price_text = page.locator(".result__price").inner_text()
        print(f"  PREDICTED: {price_text}")

        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "04_prediction_bmw.png"), full_page=True)
        print("  Screenshot: 04_prediction_bmw.png")

        # STEP 4: Test Case 3: Toyota Fortuner
        print("\n[DEMO STEP 4] Test Case 3: Toyota Fortuner, Diesel, Automatic, 5 years old")
        page.select_option("#brand", "Toyota")
        page.fill("#present_price", "32.0")
        page.fill("#kms_driven", "60000")
        page.select_option("#fuel_type", "Diesel")
        page.select_option("#seller_type", "Individual")
        page.select_option("#transmission", "Automatic")
        page.select_option("#owner", "1")
        page.fill("#car_age", "5")

        page.click("#predictBtn", force=True)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        result = page.locator(".result")
        assert result.is_visible()
        price_text = page.locator(".result__price").inner_text()
        print(f"  PREDICTED: {price_text}")

        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "05_prediction_toyota.png"), full_page=True)
        print("  Screenshot: 05_prediction_toyota.png")

        # STEP 5: Test JSON API directly
        print("\n[DEMO STEP 5] Testing JSON API endpoint...")
        import requests
        resp = requests.post(f"{live_server}/predict", json={
            "brand": "Hyundai",
            "present_price": 10.0,
            "kms_driven": 40000,
            "fuel_type": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": 0,
            "car_age": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] > 0
        print(f"  JSON API Response: {data}")

        # STEP 6: Test error handling via JSON API
        print("\n[DEMO STEP 6] Testing error handling via API (invalid price)...")
        resp = requests.post(f"{live_server}/predict", json={
            "brand": "Maruti",
            "present_price": -5,
            "kms_driven": 20000,
            "fuel_type": "Petrol",
            "seller_type": "Dealer",
            "transmission": "Manual",
            "owner": 0,
            "car_age": 3,
        })
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        data = resp.json()
        assert "error" in data
        print(f"  Error returned: '{data['error']}'")

        # Final screenshot
        page.select_option("#brand", "Hyundai")
        page.fill("#present_price", "10.0")
        page.fill("#kms_driven", "40000")
        page.select_option("#fuel_type", "Petrol")
        page.select_option("#seller_type", "Individual")
        page.select_option("#transmission", "Manual")
        page.select_option("#owner", "0")
        page.fill("#car_age", "4")
        page.click("#predictBtn", force=True)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(SCREENSHOTS_DIR, "06_prediction_hyundai.png"), full_page=True)
        print("  Screenshot: 06_prediction_hyundai.png")

        browser.close()

    print("\n" + "=" * 60)
    print("ALL E2E DEMO TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_demo_flow()
