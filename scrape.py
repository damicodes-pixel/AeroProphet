import os
import requests

# Download a smaller, manageable set of raw BTS files for notebook/model development.
# This avoids piling up a huge archive footprint while still giving a useful training sample.
years = [2024]
months = [1, 2, 3, 4, 5, 6]  # six months is enough for a baseline model run

base_url = "https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

os.makedirs("bts_raw_data", exist_ok=True)

for year in years:
    for month in months:
        file_name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        url = base_url.format(year=year, month=month)
        output_path = os.path.join("bts_raw_data", file_name)

        if os.path.exists(output_path):
            print(f"Already exists: {output_path}")
            continue

        print(f"Downloading: {year}-{month:02d}...")
        res = requests.get(url, stream=True, timeout=60)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {year}-{month}: status {res.status_code}")