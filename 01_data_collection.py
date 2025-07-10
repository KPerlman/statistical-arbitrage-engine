#!/usr/bin/env python

import time
import pandas as pd
import yfinance as yf
from typing import List


def get_sp500_tickers() -> List[str]:
    print("Fetching S&P 500 tickers from Wikipedia …")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]  # First table on the page
    tickers = (
        table["Symbol"].str.replace(".", "-", regex=False).tolist()
    )  # Replace . with - for yfinance compatibility
    print(f" - Found {len(tickers)} tickers.")
    return tickers


def download_close_prices(
    tickers: List[str], start: str, end: str, chunk_size: int = 100, delay: float = 1.0
) -> pd.DataFrame:  # Download close prices for list of tickers
    print(f"\nDownloading price data {start} → {end} …")
    all_close = pd.DataFrame()

    for i in range(0, len(tickers), chunk_size):  # Process tickers in chunks
        chunk = tickers[i : i + chunk_size]
        print(
            f"  • Chunk {i//chunk_size + 1}  ({len(chunk):3d} tickers) … ",
            end="",
            flush=True,
        )

        try:  # Download data using yfinance
            data = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )
        except Exception as exc:
            print(f"failed → {exc}")
            time.sleep(delay)
            continue

        if data.empty:
            print("no rows returned.")
            time.sleep(delay)
            continue

        if isinstance(data.columns, pd.MultiIndex):  # Check if data is multi-indexed
            close_chunk = pd.concat(  # Extract Close prices from DataFrame
                {tkr: data[tkr]["Close"] for tkr in chunk if tkr in data}, axis=1
            )
        else:  # If not multi-indexed assume single index
            close_chunk = data[["Close"]].rename(columns={"Close": chunk[0]})

        dead = close_chunk.columns[close_chunk.isna().all()].tolist()
        if dead:  # Check for tickers with all NaN values
            print(f"Warning:  {len(dead)} failed: {dead}")
        else:
            print("OK")

        all_close = pd.concat(
            [all_close, close_chunk], axis=1
        )  # Append chunk to all_close DataFrame
        time.sleep(delay)

    return all_close


if __name__ == "__main__":
    START_DATE = "2019-01-01"
    END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
    OUTPUT_CSV = "sp500_close_prices.csv"

    sp500 = get_sp500_tickers()
    close = download_close_prices(sp500, START_DATE, END_DATE)

    if close.empty:
        raise RuntimeError("No data downloaded – script aborting.")

    print(f"\nDownloaded dataframe shape (before cleaning): {close.shape}")

    # Remove columns with too many NaN values
    thresh = int(0.9 * len(close))
    close = close.dropna(axis="columns", thresh=thresh)
    close.ffill(inplace=True)
    close.bfill(inplace=True)

    print(f"Shape after cleaning: {close.shape}")
    close.to_csv(OUTPUT_CSV)
    print(f"\n Saved cleaned data to “{OUTPUT_CSV}”.")
