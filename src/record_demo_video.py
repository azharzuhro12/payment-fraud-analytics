"""
Records a real browser walkthrough of the running Streamlit app (http://localhost:8501)
using Playwright + Google Chrome, capturing frames across:
1. Tab 1 (Analytics Dashboard): Top KPIs & Channel/Hourly charts -> Scrolling down through Category Treemap,
   Feature Importance, 4-Cluster PCA Scatter, and Priority Merchant Watchlist.
2. Tab 2 (Real-Time Fraud Prediction): Clicking Preset Scenarios, Evaluating Transaction Risk
   (showing Speedometer Gauge & Decision), and Scoring Live Batch Stream.
Outputs both:
- reports/figures/demo.gif (Auto-playing embedded video animation in GitHub README)
- reports/figures/demo.mp4 (H.264 MP4 video file)
"""

import os
import time
import subprocess
from PIL import Image
from playwright.sync_api import sync_playwright

FRAMES_DIR = "reports/figures/_frames"
GIF_OUT = "reports/figures/demo.gif"
MP4_OUT = "reports/figures/demo.mp4"


def scroll_streamlit(page, y_pos: int):
    page.evaluate(f"""
        const main = document.querySelector('section.main') || document.querySelector('[data-testid="stAppViewContainer"]') || window;
        main.scrollTo({{top: {y_pos}, behavior: 'instant'}});
        window.scrollTo({{top: {y_pos}, behavior: 'instant'}});
    """)


def capture_demo():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))

    unique_paths = []
    unique_durations = []

    def snap(page, hold_ms=1400):
        idx = len(unique_paths)
        path = os.path.join(FRAMES_DIR, f"frame_{idx:03d}.png")
        page.screenshot(path=path)
        unique_paths.append(path)
        unique_durations.append(hold_ms)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--window-size=1440,900"]
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Opening http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        time.sleep(3.5)

        # 1. Tab 1 Top: KPIs + Insight 1 & 2
        snap(page, hold_ms=1600)

        # 2. Scroll to Row 2: Treemap + Root Cause Feature Importance
        h3 = page.get_by_text("3️⃣ Financial Fraud Loss Exposure")
        if h3.count() > 0:
            h3.first.scroll_into_view_if_needed()
            time.sleep(1.2)
            snap(page, hold_ms=1600)

        # Click sub-tab "💰 Financial ROI & Threshold Optimization"
        roi_tab = page.get_by_text("💰 Financial ROI & Threshold Optimization")
        if roi_tab.count() > 0:
            roi_tab.first.click()
            time.sleep(1.2)
            snap(page, hold_ms=1600)

        # 3. Scroll to Row 3: K-Means 4-Cluster PCA Scatter & Segment Table
        h5 = page.get_by_text("5️⃣ Merchant Behavioral Segmentation")
        if h5.count() > 0:
            h5.first.scroll_into_view_if_needed()
            time.sleep(1.2)
            snap(page, hold_ms=1600)

        # 4. Scroll to Priority Merchant Audit Watchlist
        hw = page.get_by_text("🚨 Priority Merchant Audit Watchlist")
        if hw.count() > 0:
            hw.first.scroll_into_view_if_needed()
            time.sleep(1.2)
            snap(page, hold_ms=1600)

        # 5. Switch to Tab 2: Real-Time Fraud Prediction
        tab2 = page.get_by_text("🎯 Real-Time Fraud Prediction")
        if tab2.count() > 0:
            tab2.first.scroll_into_view_if_needed()
            time.sleep(0.5)
            tab2.first.click()
            time.sleep(1.8)
            snap(page, hold_ms=1500)

            # Click Scenario 1: Normal Retail POS (Approve) -> Evaluate
            s1 = page.get_by_text("✅ Scenario 1: Normal Retail POS")
            if s1.count() > 0:
                s1.first.click()
                time.sleep(1.5)
            ev = page.get_by_text("Evaluate Transaction Risk")
            if ev.count() > 0:
                ev.first.click()
                time.sleep(2.2)
                dec = page.get_by_text("Risk Evaluation Decision")
                if dec.count() > 0:
                    dec.first.scroll_into_view_if_needed()
                    time.sleep(1.0)
                snap(page, hold_ms=1800)

            # Click Scenario 3: Account Takeover / EMV Fallback (Auto-Block) -> Evaluate
            s3 = page.get_by_text("🚨 Scenario 3: Account Takeover")
            if s3.count() > 0:
                s3.first.scroll_into_view_if_needed()
                time.sleep(0.5)
                s3.first.click()
                time.sleep(1.5)
            ev = page.get_by_text("Evaluate Transaction Risk")
            if ev.count() > 0:
                ev.first.click()
                time.sleep(2.2)
                dec = page.get_by_text("Risk Evaluation Decision")
                if dec.count() > 0:
                    dec.first.scroll_into_view_if_needed()
                    time.sleep(1.0)
                snap(page, hold_ms=1900)

            # Click Live Batch Transaction Stream Scorer
            batch_btn = page.get_by_text("🔄 Score Sample Live Stream")
            if batch_btn.count() > 0:
                batch_btn.first.scroll_into_view_if_needed()
                time.sleep(0.6)
                batch_btn.first.click()
                time.sleep(2.5)
                batch_btn.first.scroll_into_view_if_needed()
                time.sleep(1.0)
                snap(page, hold_ms=2200)

        browser.close()

    print(f"Captured {len(unique_paths)} distinct workflow screens. Building animated GIF and MP4...")

    pil_frames = []
    for p_img in unique_paths:
        im = Image.open(p_img).convert("RGB")
        im = im.resize((1200, 750), Image.Resampling.LANCZOS)
        pil_frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    pil_frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=pil_frames[1:],
        duration=unique_durations,
        loop=0,
        optimize=True
    )
    print(f"Saved animated GIF: {GIF_OUT}")

    ffmpeg_bin = "/home/zarr/.local/bin/ffmpeg"
    if os.path.exists(ffmpeg_bin):
        subprocess.run([
            ffmpeg_bin, "-y", "-i", GIF_OUT,
            "-movflags", "faststart", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            MP4_OUT
        ], check=False)
        print(f"Saved MP4 video: {MP4_OUT}")

    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))
    os.rmdir(FRAMES_DIR)


if __name__ == "__main__":
    capture_demo()
