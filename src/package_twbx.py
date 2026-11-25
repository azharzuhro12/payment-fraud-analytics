"""
Creates a .twbx (Tableau Packaged Workbook) which is a ZIP archive containing:
1. payment_fraud_dashboard.twb (XML definition)
2. Data/ (Embedded CSV datasets)
This format is 100% self-contained and universally accepted by Tableau Web Upload & Desktop.
"""

import os
import zipfile


def create_twbx():
    base_dir = os.path.abspath("data/tableau")
    twb_file = os.path.join(base_dir, "payment_fraud_dashboard.twb")
    csv_channel = os.path.join(base_dir, "channel_location_risk.csv")
    csv_hourly = os.path.join(base_dir, "hourly_fraud_risk.csv")
    csv_segments = os.path.join(base_dir, "merchant_segments.csv")
    twbx_file = os.path.join(base_dir, "payment_fraud_dashboard.twbx")

    with zipfile.ZipFile(twbx_file, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(twb_file, arcname="payment_fraud_dashboard.twb")
        z.write(csv_channel, arcname="Data/channel_location_risk.csv")
        z.write(csv_hourly, arcname="Data/hourly_fraud_risk.csv")
        z.write(csv_segments, arcname="Data/merchant_segments.csv")

    print(f"Successfully generated packaged workbook: {twbx_file}")


if __name__ == "__main__":
    create_twbx()
