import os
import requests

# Download 2 years of data (2024–2025) for a balanced historical graph
years = [2024, 2025]
months = range(1, 13)

base_url = "https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

os.makedirs("bts_raw_data", exist_ok=True)

for year in years:
    for month in months:
        file_name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        url = base_url.format(year=year, month=month)
        output_path = os.path.join("bts_raw_data", file_name)
        
        print(f"Downloading: {year}-{month:02d}...")
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        else:
            print(f"Failed to download {year}-{month}")