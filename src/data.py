from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "TLT", "GLD"] #the file loads the data, these are the tickers, snp 500,long term treasury, and gold. 
#fetch downloads the data from yahoo finance, load loads the data from the raw folder, and main downloads the data and saves it to the raw folder.
RAW = Path("data/raw")


def fetch(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    df.index = df.index.tz_localize(None).normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "date"
    return df


def load(ticker: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{ticker}.csv", index_col="date", parse_dates=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        df = fetch(t)
        df.to_csv(RAW / f"{t}.csv")
        print(t, df.index.min().date(), df.index.max().date(), len(df))


if __name__ == "__main__":
    main()