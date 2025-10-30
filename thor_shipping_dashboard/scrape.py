from typing import Optional
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_ayn_dashboard(url: str = "https://www.ayntec.com/pages/shipment-dashboard") -> pd.DataFrame:
    """Scrape AYN shipment dashboard and return normalized DataFrame.

    The page lists sections per date like "2025/10/13" followed by lines such as
    "AYN Thor Black Pro: 732xx--736xx". This function parses those blocks and
    returns a DataFrame with columns: ``date``, ``variant``, ``color``,
    ``start_prefix``, ``end_prefix``.
    """
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n")

    date_pattern = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")
    line_pattern = re.compile(
        r"AYN\s+Thor\s+(?P<color>[A-Za-z ]+?)\s+(?P<variant>Lite|Base|Pro|Max)\s*:\s*(?P<start>\d{3,5})xx\s*--\s*(?P<end>\d{3,5})xx",
        re.IGNORECASE,
    )

    rows: list[dict[str, object]] = []
    current_date: Optional[str] = None
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        mdate = date_pattern.match(raw_line)
        if mdate:
            current_date = f"{mdate.group(1)}-{mdate.group(2)}-{mdate.group(3)}"
            continue
        mline = line_pattern.search(raw_line)
        if current_date and mline:
            color = mline.group("color").strip()
            color = " ".join(p.capitalize() for p in color.split())
            variant = mline.group("variant").capitalize()
            start_prefix = int(mline.group("start")[:3])
            end_prefix = int(mline.group("end")[:3])
            rows.append(
                {
                    "date": current_date,
                    "variant": variant,
                    "color": color,
                    "start_prefix": start_prefix,
                    "end_prefix": end_prefix,
                }
            )

    if not rows:
        raise ValueError("No shipping rows parsed from AYN dashboard")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=False)
    df["start_prefix"] = df["start_prefix"].astype(int)
    df["end_prefix"] = df["end_prefix"].astype(int)
    return df


