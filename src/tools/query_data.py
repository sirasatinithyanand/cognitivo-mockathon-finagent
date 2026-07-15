"""`query_data` — precise structured queries over RBA rates, ASX prices, and AFR news.

Reads JSONL files directly — no database required.
Set DATA_DIR env var to point at your dataset directory.

Supports:
  dataset="rba"  → RBA cash rate decisions (175 records, 2010-2026)
  dataset="asx"  → ASX OHLCV prices (18 tickers, 2015-2021, 1774 rows each)
  dataset="afr"  → AFR news articles (86 monthly files, 2015-2021)
"""
from __future__ import annotations

import glob
import json
import os
import re as _re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from .registry import register

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/home/cognitivo/Downloads/Jasonl format DataSets"
)
RBA_PATH = os.path.join(DATA_DIR, "RBA-Rates-2010-2026", "RBA-rates.jsonl")
ASX_DIR  = os.path.join(DATA_DIR, "ASX-18-companies-2015-2021-Jasonl")
AFR_DIR  = os.path.join(DATA_DIR, "AFR Jasonl")

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset": {
            "type": "string",
            "enum": ["rba", "asx", "afr"],
            "description": "Dataset: 'rba' for cash-rate decisions, 'asx' for stock prices, 'afr' for news articles",
        },
        "metric": {
            "type": "string",
            "description": (
                "RBA: count, list, sum_change, lookup_rate, count_changes, count_increases, count_decreases, count_holds, change_distribution, extremes, max_hold_streak, min_change_interval, count_by_year, post_cut_basket_returns, dataset_info. "
                "ASX: count, list, lookup_price, annual_return, rank_annual_returns, full_sample_return, volatility, max_volume, avg_close, avg_volume, "
                "max_drawdown, price_return_between, correlation, cross_year_extremes, dataset_info. "
                "AFR: count, count_by_month, share."
            ),
        },
        "ticker": {
            "type": "string",
            "description": "ASX ticker e.g. 'BHP.AX'. Use 'all' for all 18 tickers (excl Tabcorp filter handled manually).",
        },
        "pattern": {
            "type": "string",
            "description": "Regex pattern for AFR text search (case-insensitive). e.g. '\\\\bQBE\\\\b' or 'royal commission'.",
        },
        "date_from": {"type": "string", "description": "Start date inclusive, ISO format YYYY-MM-DD"},
        "date_to":   {"type": "string", "description": "End date inclusive, ISO format YYYY-MM-DD"},
        "year":      {"type": "integer", "description": "Shorthand for a full calendar year (sets date_from/date_to)"},
        "change":    {
            "type": "string",
            "enum": ["increase", "decrease", "any", "none"],
            "description": "RBA filter: 'increase', 'decrease', 'any' (changed), 'none' (unchanged)",
        },
        "limit":     {"type": "integer", "description": "Max rows to return in list/rank modes (default 20)"},
        "exclude_tickers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ASX tickers to exclude, e.g. ['TAH.AX'] to exclude Tabcorp",
        },
    },
    "required": ["dataset", "metric"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {},
        "records": {"type": "array"},
        "meta": {"type": "object"},
    },
}


# ── RBA helpers ───────────────────────────────────────────────────────────────

def _load_rba() -> list[dict]:
    records = []
    try:
        with open(RBA_PATH, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                try:
                    d = datetime.strptime(r["Effective Date"].strip(), "%d %b %Y").date()
                except ValueError:
                    d = datetime.strptime(r["Effective Date"].strip(), "%d %B %Y").date()
                change = float(r["Change % points"].replace("+", "") or "0")
                rate   = float(r["Cash rate target%"])
                records.append({"date": d, "change": change, "rate": rate})
    except FileNotFoundError:
        return []
    return sorted(records, key=lambda r: r["date"])


def _filter_rba(records, date_from=None, date_to=None, change=None):
    df = _parse_date(date_from)
    dt = _parse_date(date_to)
    out = []
    for r in records:
        if df and r["date"] < df:
            continue
        if dt and r["date"] > dt:
            continue
        if change == "increase" and r["change"] <= 0:
            continue
        if change == "decrease" and r["change"] >= 0:
            continue
        if change == "any" and r["change"] == 0:
            continue
        if change == "none" and r["change"] != 0:
            continue
        out.append(r)
    return out


def _rba_to_json(r: dict) -> dict:
    from datetime import timedelta
    d = r["date"]
    return {"date": str(d), "change": r["change"], "rate": r["rate"],
            "week_after": str(d + timedelta(days=7))}


# ── ASX helpers ───────────────────────────────────────────────────────────────

_ASX_TICKER_TO_FILE: dict[str, str] | None = None

def _build_asx_ticker_map() -> dict[str, str]:
    global _ASX_TICKER_TO_FILE
    if _ASX_TICKER_TO_FILE is not None:
        return _ASX_TICKER_TO_FILE
    mapping: dict[str, str] = {}
    for path in glob.glob(os.path.join(ASX_DIR, "*.jsonl")):
        with open(path, encoding="utf-8-sig") as f:
            first = f.readline().strip()
            if first:
                try:
                    ticker = json.loads(first).get("ticker", "")
                    if ticker:
                        mapping[ticker] = path
                except Exception:
                    pass
    _ASX_TICKER_TO_FILE = mapping
    return mapping


def _load_asx(ticker: str = "all") -> list[dict]:
    records = []
    if ticker and ticker != "all":
        mapping = _build_asx_ticker_map()
        paths = [mapping[ticker]] if ticker in mapping else []
        if not paths:
            # fallback: glob by base name prefix
            base = ticker.split(".")[0]
            paths = glob.glob(os.path.join(ASX_DIR, f"*{base}*"))
    else:
        paths = glob.glob(os.path.join(ASX_DIR, "*.jsonl"))

    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["date"] = date.fromisoformat(r["date"])
                records.append(r)
    return sorted(records, key=lambda r: (r["ticker"], r["date"]))


def _filter_asx(records, date_from=None, date_to=None, ticker=None, exclude=None):
    df = _parse_date(date_from)
    dt = _parse_date(date_to)
    exclude = set(exclude or [])
    out = []
    for r in records:
        if ticker and ticker != "all" and r["ticker"] != ticker:
            continue
        if r["ticker"] in exclude:
            continue
        if df and r["date"] < df:
            continue
        if dt and r["date"] > dt:
            continue
        out.append(r)
    return out


def _annual_return(records, ticker, year) -> float | None:
    rows = [r for r in records if r["ticker"] == ticker and r["date"].year == year]
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r["date"])
    start, end = rows[0]["close"], rows[-1]["close"]
    return round((end - start) / start * 100, 4) if start else None


def _asx_to_json(r: dict) -> dict:
    return {k: str(v) if isinstance(v, date) else v for k, v in r.items()}


# ── AFR helpers ───────────────────────────────────────────────────────────────

def _iter_afr(date_from=None, date_to=None):
    """Yield AFR records filtered by date range."""
    df = _parse_date(date_from)
    dt = _parse_date(date_to)
    for fn in sorted(glob.glob(os.path.join(AFR_DIR, "AFR_*.jsonl"))):
        # Fast skip: parse year from filename
        base = os.path.basename(fn)
        try:
            file_year = int(base[4:8])
        except ValueError:
            pass
        else:
            if df and file_year < df.year:
                continue
            if dt and file_year > dt.year:
                continue
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pub = str(r.get("PUBLICATIONDATE", ""))
                if len(pub) < 6:
                    continue
                try:
                    y, m, d_val = int(pub[:4]), int(pub[4:6]), int(pub[6:8]) if len(pub) >= 8 else 1
                    rec_date = date(y, m, d_val)
                except (ValueError, TypeError):
                    continue
                if df and rec_date < df:
                    continue
                if dt and rec_date > dt:
                    continue
                yield r, rec_date


def _afr_text(r: dict) -> str:
    return " ".join(str(r.get(k, "")) for k in ["HEADLINE", "SUBHEAD", "INTRO", "TEXT"])


# ── Shared ────────────────────────────────────────────────────────────────────

def _parse_date(s) -> date | None:
    if not s:
        return None
    return date.fromisoformat(str(s))


# ── Tool ──────────────────────────────────────────────────────────────────────

@register(
    "query_data",
    (
        "Query Australian financial datasets directly. "
        "dataset='rba': RBA cash rate decisions — counts, rate values, change history. "
        "dataset='asx': ASX stock OHLCV — close prices, annual returns, rankings, avg volume, drawdowns, date-range returns. "
        "dataset='afr': AFR news — regex pattern counts and rankings by year/month. "
        "Use exclude_tickers=['TAH.AX'] to exclude Tabcorp from ASX queries."
    ),
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)
def query_data(
    dataset: str,
    metric: str,
    ticker: str | None = None,
    pattern: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    year: int | None = None,
    change: str | None = None,
    limit: int = 100,
    exclude_tickers: list | None = None,
) -> dict:
    # Year shorthand
    if year:
        date_from = date_from or f"{year}-01-01"
        date_to   = date_to   or f"{year}-12-31"

    # ── RBA ──────────────────────────────────────────────────────────────────
    if dataset == "rba":
        all_records = _load_rba()
        if not all_records:
            return {"error": f"RBA data not found at {RBA_PATH}. Set DATA_DIR env var."}

        if metric == "dataset_info":
            return {
                "total_records": len(all_records),
                "first_record": _rba_to_json(all_records[0]),
                "last_record": _rba_to_json(all_records[-1]),
                "date_from": str(all_records[0]["date"]),
                "date_to": str(all_records[-1]["date"]),
                "note": "175 decision records, 2010-02-03 to 2026-06-17",
            }

        filtered = _filter_rba(all_records, date_from, date_to, change)

        if metric == "count":
            return {"result": len(filtered), "meta": {"total": len(all_records), "filtered": len(filtered)}}

        if metric == "count_changes":
            ch = change if change in ("increase", "decrease", "any") else "any"
            filtered_changes = _filter_rba(all_records, date_from, date_to, ch)
            increases = len(_filter_rba(all_records, date_from, date_to, "increase"))
            decreases = len(_filter_rba(all_records, date_from, date_to, "decrease"))
            return {
                "result": len(filtered_changes),
                "increases": increases,
                "decreases": decreases,
                "meta": {"total": len(all_records)},
            }

        if metric == "count_increases":
            recs = _filter_rba(all_records, date_from, date_to, "increase")
            by_year = {}
            for r in recs:
                by_year[r["date"].year] = by_year.get(r["date"].year, 0) + 1
            sorted_recs = sorted(recs, key=lambda r: r["date"])
            total_change = round(sum(r["change"] for r in sorted_recs), 4) if sorted_recs else None
            rate_before = round(sorted_recs[0]["rate"] - sorted_recs[0]["change"], 4) if sorted_recs else None
            return {
                "result": len(recs),
                "total_change_pp": total_change,
                "rate_before_first_hike": rate_before,
                "by_year": dict(sorted(by_year.items())),
                "first_date": str(sorted_recs[0]["date"]) if sorted_recs else None,
                "last_date": str(sorted_recs[-1]["date"]) if sorted_recs else None,
            }

        if metric == "count_decreases":
            recs = _filter_rba(all_records, date_from, date_to, "decrease")
            by_year = {}
            for r in recs:
                by_year[r["date"].year] = by_year.get(r["date"].year, 0) + 1
            sorted_recs = sorted(recs, key=lambda r: r["date"])
            total_change = round(sum(r["change"] for r in sorted_recs), 4) if sorted_recs else None
            rate_before = round(sorted_recs[0]["rate"] - sorted_recs[0]["change"], 4) if sorted_recs else None
            rate_after = sorted_recs[-1]["rate"] if sorted_recs else None
            return {
                "result": len(recs),
                "total_change_pp": total_change,
                "rate_before_first_cut": rate_before,
                "rate_after_last_cut": rate_after,
                "by_year": dict(sorted(by_year.items())),
                "first_date": str(sorted_recs[0]["date"]) if sorted_recs else None,
                "last_date": str(sorted_recs[-1]["date"]) if sorted_recs else None,
            }

        if metric == "sum_change":
            if not date_from and not date_to:
                return {"error": "sum_change requires date_from and/or date_to. Use count_decreases/count_increases with date filters instead — they now include total_change_pp, rate_before_first_cut, and rate_after_last_cut."}
            total = round(sum(r["change"] for r in filtered), 4)
            sorted_f = sorted(filtered, key=lambda r: r["date"])
            rate_before = round(sorted_f[0]["rate"] - sorted_f[0]["change"], 4) if sorted_f else None
            rate_after = sorted_f[-1]["rate"] if sorted_f else None
            return {
                "result": total, "unit": "percentage_points",
                "rate_before_first": rate_before, "rate_after_last": rate_after,
                "meta": {"n_records": len(filtered)},
            }

        if metric == "count_by_year":
            from collections import Counter
            year_counts = Counter(r["date"].year for r in filtered)
            return {"by_year": dict(sorted(year_counts.items())), "total": len(filtered)}

        if metric == "post_cut_basket_returns":
            from datetime import timedelta
            cuts = sorted(_filter_rba(all_records, date_from, date_to, "decrease"), key=lambda r: r["date"])
            if not cuts:
                return {"error": "No cuts found in specified period"}
            all_asx = _load_asx("all")
            excl_set = set(exclude_tickers or [])
            results = []
            for cut in cuts:
                df_d = cut["date"]
                dt_d = df_d + timedelta(days=7)
                by_ticker: dict[str, list] = defaultdict(list)
                for r in all_asx:
                    if r["ticker"] in excl_set:
                        continue
                    if r["date"] >= df_d and r["date"] <= dt_d:
                        by_ticker[r["ticker"]].append(r)
                ticker_rets = []
                for tk, recs in by_ticker.items():
                    recs.sort(key=lambda r: r["date"])
                    r1 = next((r for r in recs if r["date"] >= df_d), None)
                    r2 = next((r for r in recs if r["date"] >= dt_d), None)
                    if r1 and r2 and r1 is not r2:
                        ret = round((r2["close"] - r1["close"]) / r1["close"] * 100, 4)
                        ticker_rets.append({"ticker": tk, "return_pct": ret})
                basket = round(sum(t["return_pct"] for t in ticker_rets) / len(ticker_rets), 2) if ticker_rets else None
                results.append({
                    "cut_date": str(df_d), "new_rate": cut["rate"],
                    "end_date": str(dt_d),
                    "basket_return_pct": basket,
                    "n_tickers": len(ticker_rets),
                })
            return {"cuts": results, "meta": {"date_from": date_from, "date_to": date_to}}

        if metric == "lookup_rate":
            target = _parse_date(date_from)
            exact = [r for r in all_records if r["date"] == target]
            if exact:
                return {"result": exact[0]["rate"], "date": str(target)}
            # Return the rate in effect ON OR BEFORE the target date (not nearest which could be future)
            before = [r for r in all_records if r["date"] < target]
            if before:
                rec = before[-1]  # most recent record before target
                return {"result": rec["rate"], "date": str(rec["date"]), "note": "rate in effect on target date"}
            after = [r for r in all_records if r["date"] > target]
            if after:
                return {"result": after[0]["rate"], "date": str(after[0]["date"]), "note": "earliest record after target date"}
            return {"error": "No RBA records found"}

        if metric == "extremes":
            # Min and max cash-rate target across all (or filtered) records
            recs = _filter_rba(all_records, date_from, date_to)
            if not recs:
                return {"error": "No records found"}
            min_rate = min(r["rate"] for r in recs)
            max_rate = max(r["rate"] for r in recs)
            min_recs = sorted([r for r in recs if r["rate"] == min_rate], key=lambda r: r["date"])
            max_recs = sorted([r for r in recs if r["rate"] == max_rate], key=lambda r: r["date"])
            return {
                "min_rate": min_rate,
                "min_first_date": str(min_recs[0]["date"]),
                "min_record_count": len(min_recs),
                "max_rate": max_rate,
                "max_first_date": str(max_recs[0]["date"]),
                "max_record_count": len(max_recs),
            }

        if metric == "count_holds":
            all_filtered = _filter_rba(all_records, date_from, date_to)
            holds = [r for r in all_filtered if r["change"] == 0]
            total = len(all_filtered)
            pct = round(len(holds) / total * 100, 4) if total else 0
            # per-year breakdown and years where every decision was a hold
            by_year: dict[int, dict] = {}
            for r in all_filtered:
                yr = r["date"].year
                if yr not in by_year:
                    by_year[yr] = {"total": 0, "holds": 0}
                by_year[yr]["total"] += 1
                if r["change"] == 0:
                    by_year[yr]["holds"] += 1
            no_change_years = [yr for yr, c in sorted(by_year.items()) if c["holds"] == c["total"]]
            return {"result": len(holds), "total": total, "pct_of_total": pct,
                    "no_change_years": no_change_years, "by_year": dict(sorted(by_year.items()))}

        if metric == "change_distribution":
            # Frequency of each non-zero change size
            changes = _filter_rba(all_records, date_from, date_to, "any")
            freq: dict[float, int] = {}
            for r in changes:
                freq[r["change"]] = freq.get(r["change"], 0) + 1
            ranked = sorted(freq.items(), key=lambda x: -x[1])
            return {
                "distribution": [{"change_pp": k, "count": v} for k, v in ranked],
                "most_common": {"change_pp": ranked[0][0], "count": ranked[0][1]} if ranked else None,
                "total_non_hold": len(changes),
            }

        if metric == "max_hold_streak":
            # Longest gap in days between two consecutive non-zero changes
            changes = sorted(_filter_rba(all_records, date_from, date_to, "any"), key=lambda r: r["date"])
            if len(changes) < 2:
                return {"error": "Need at least 2 non-zero changes"}
            best = {"days": 0}
            for i in range(1, len(changes)):
                days = (changes[i]["date"] - changes[i-1]["date"]).days
                if days > best["days"]:
                    best = {
                        "days": days,
                        "from_date": str(changes[i-1]["date"]),
                        "from_rate": changes[i-1]["rate"],
                        "to_date": str(changes[i]["date"]),
                        "to_rate": changes[i]["rate"],
                    }
            return best

        if metric == "min_change_interval":
            # Shortest gap in days between two consecutive non-zero changes
            changes = sorted(_filter_rba(all_records, date_from, date_to, "any"), key=lambda r: r["date"])
            if len(changes) < 2:
                return {"error": "Need at least 2 non-zero changes"}
            best = {"days": 999999}
            for i in range(1, len(changes)):
                days = (changes[i]["date"] - changes[i-1]["date"]).days
                if days < best["days"]:
                    best = {
                        "days": days,
                        "from_date": str(changes[i-1]["date"]),
                        "from_rate": changes[i-1]["rate"],
                        "to_date": str(changes[i]["date"]),
                        "to_rate": changes[i]["rate"],
                    }
            return best

        if metric == "list":
            rows = [_rba_to_json(r) for r in filtered[:limit]]
            return {"records": rows, "meta": {"total_filtered": len(filtered), "shown": len(rows)}}

        return {"error": f"Unknown metric '{metric}' for rba."}

    # ── ASX ──────────────────────────────────────────────────────────────────
    if dataset == "asx":
        all_tickers = sorted(_build_asx_ticker_map().keys())
        if not all_tickers:
            return {"error": f"ASX data not found at {ASX_DIR}. Set DATA_DIR env var."}

        t = ticker if ticker and ticker != "all" else None
        excl = list(exclude_tickers or [])

        if metric == "dataset_info":
            all_asx = _load_asx("all")
            dates = [r["date"] for r in all_asx]
            return {
                "tickers": all_tickers,
                "n_tickers": len(all_tickers),
                "rows_per_ticker": len(all_asx) // len(all_tickers),
                "date_from": str(min(dates)),
                "date_to": str(max(dates)),
                "total_rows": len(all_asx),
            }

        if metric == "rank_annual_returns":
            yr = year or (int(date_from[:4]) if date_from else 2018)
            all_asx = _load_asx("all")
            ranks = []
            for tk in all_tickers:
                if tk in excl:
                    continue
                ret = _annual_return(all_asx, tk, yr)
                if ret is not None:
                    ranks.append({"ticker": tk, "annual_return_pct": ret})
            ranks.sort(key=lambda x: x["annual_return_pct"], reverse=True)
            avg = round(sum(r["annual_return_pct"] for r in ranks) / len(ranks), 4) if ranks else None
            return {
                "best": ranks[0] if ranks else None,
                "worst": ranks[-1] if ranks else None,
                "top5": ranks[:5],
                "bottom3": ranks[-3:],
                "all_tickers": ranks,
                "avg_return_pct": avg,
                "meta": {"year": yr, "n_tickers": len(ranks)},
            }

        if metric == "annual_return":
            yr = year or (int(date_from[:4]) if date_from else None)
            if not yr:
                return {"error": "Provide year or date_from for annual_return"}
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, ticker=t, exclude=excl)
            result = {}
            for tk in ({t} if t else set(r["ticker"] for r in filtered) - set(excl)):
                ret = _annual_return(filtered, tk, yr)
                result[tk] = ret
            avg = round(sum(v for v in result.values() if v is not None) / max(1, sum(1 for v in result.values() if v is not None)), 4)
            return {"result": result, "basket_avg_return_pct": avg, "year": yr}

        if metric == "lookup_price":
            if not t:
                return {"error": "Provide ticker for lookup_price"}
            target = _parse_date(date_from)
            rows = _load_asx(t)
            exact = [r for r in rows if r["date"] == target]
            if exact:
                return {"result": _asx_to_json(exact[0])}
            if not rows:
                return {"error": f"No data for {t}"}
            nearest = min(rows, key=lambda r: abs((r["date"] - target).days))
            return {"result": _asx_to_json(nearest), "note": "nearest trading day"}

        if metric == "avg_close":
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            if not filtered:
                return {"result": None, "error": "No records matched"}
            avg = round(sum(r["close"] for r in filtered) / len(filtered), 4)
            return {"result": avg, "n_records": len(filtered)}

        if metric == "avg_volume":
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            if not filtered:
                return {"result": None, "error": "No records matched"}
            by_ticker: dict[str, list] = defaultdict(list)
            for r in filtered:
                by_ticker[r["ticker"]].append(r["volume"])
            avgs = [
                {"ticker": tk, "avg_volume": round(sum(vols) / len(vols), 2)}
                for tk, vols in by_ticker.items()
            ]
            avgs.sort(key=lambda x: -x["avg_volume"])
            if t:
                return {"result": avgs[0]["avg_volume"] if avgs else None, "ticker": t}
            # Return compact ranking — top 5 only to keep response small
            return {"best": avgs[0] if avgs else None, "top5": avgs[:5], "meta": {"n_tickers": len(avgs)}}

        if metric == "max_drawdown":
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            by_ticker: dict[str, list] = defaultdict(list)
            for r in filtered:
                by_ticker[r["ticker"]].append(r)
            results = []
            for tk, recs in by_ticker.items():
                recs.sort(key=lambda r: r["date"])
                peak_price = recs[0]["close"]
                peak_date = recs[0]["date"]
                max_dd = 0.0
                trough_date = peak_date
                cur_peak = peak_price
                cur_peak_date = peak_date
                for r in recs:
                    if r["close"] > cur_peak:
                        cur_peak = r["close"]
                        cur_peak_date = r["date"]
                    dd = (r["close"] - cur_peak) / cur_peak * 100
                    if dd < max_dd:
                        max_dd = dd
                        trough_date = r["date"]
                        peak_date = cur_peak_date
                        peak_price = cur_peak
                results.append({
                    "ticker": tk,
                    "max_drawdown_pct": round(max_dd, 4),
                    "peak_date": str(peak_date),
                    "trough_date": str(trough_date),
                    "peak_price": round(peak_price, 4),
                })
            results.sort(key=lambda x: x["max_drawdown_pct"])  # most negative first
            best3 = list(reversed(results[-3:]))  # least negative first (smallest loss first)
            return {"worst3": results[:3], "worst": results[0] if results else None,
                    "best3": best3, "best": results[-1] if results else None,
                    "all_tickers": results,
                    "meta": {"n_tickers": len(results)}}

        if metric == "price_return_between":
            df_d = _parse_date(date_from)
            dt_d = _parse_date(date_to)
            if not df_d or not dt_d:
                return {"error": "Provide date_from and date_to for price_return_between"}
            rows = _load_asx(t or "all")
            by_ticker: dict[str, list] = defaultdict(list)
            for r in _filter_asx(rows, ticker=t, exclude=excl):
                by_ticker[r["ticker"]].append(r)
            results = []
            for tk, recs in by_ticker.items():
                recs.sort(key=lambda r: r["date"])
                r1 = next((r for r in recs if r["date"] >= df_d), None)
                r2 = next((r for r in recs if r["date"] >= dt_d), None)
                if r1 and r2 and r1 is not r2:
                    ret = round((r2["close"] - r1["close"]) / r1["close"] * 100, 4)
                    results.append({"ticker": tk, "return_pct": ret,
                                    "from_date": str(r1["date"]), "to_date": str(r2["date"])})
            results.sort(key=lambda x: x["ticker"])
            basket_avg = round(sum(r["return_pct"] for r in results) / len(results), 4) if results else None
            return {
                "basket_avg_return_pct": basket_avg,
                "per_ticker": results,  # compact — just ticker + return_pct
                "meta": {"n_tickers": len(results), "from": str(df_d), "to": str(dt_d)},
            }

        if metric == "full_sample_return":
            # First-to-last return across entire dataset (2015-01-02 to 2021-12-30)
            rows = _load_asx(t or "all")
            by_ticker: dict[str, list] = defaultdict(list)
            for r in _filter_asx(rows, ticker=t, exclude=excl):
                by_ticker[r["ticker"]].append(r)
            results = []
            for tk, recs in by_ticker.items():
                recs.sort(key=lambda r: r["date"])
                start, end = recs[0]["close"], recs[-1]["close"]
                ret = round((end - start) / start * 100, 4) if start else None
                results.append({
                    "ticker": tk, "return_pct": ret,
                    "start_price": round(start, 4), "start_date": str(recs[0]["date"]),
                    "end_price": round(end, 4), "end_date": str(recs[-1]["date"]),
                })
            results.sort(key=lambda x: x["return_pct"] if x["return_pct"] is not None else -999, reverse=True)
            valid_rets = sorted([r["return_pct"] for r in results if r["return_pct"] is not None])
            avg = round(sum(valid_rets) / len(valid_rets), 4) if valid_rets else None
            n = len(valid_rets)
            median = round((valid_rets[n//2] if n % 2 == 1 else (valid_rets[n//2-1] + valid_rets[n//2]) / 2), 4) if n else None
            positive_count = sum(1 for r in valid_rets if r > 0)
            negative_count = sum(1 for r in valid_rets if r < 0)
            return {
                "best": results[0] if results else None,
                "worst": results[-1] if results else None,
                "all_tickers": results,
                "avg_return_pct": avg,
                "median_return_pct": median,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "meta": {"n_tickers": len(results)},
            }

        if metric == "volatility":
            # Annualised volatility = std dev of daily log returns × sqrt(252)
            import math
            rows = _load_asx(t or "all")
            by_ticker: dict[str, list] = defaultdict(list)
            for r in _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl):
                by_ticker[r["ticker"]].append(r)
            results = []
            for tk, recs in by_ticker.items():
                recs.sort(key=lambda r: r["date"])
                log_rets = []
                for i in range(1, len(recs)):
                    p0, p1 = recs[i-1]["close"], recs[i]["close"]
                    if p0 > 0 and p1 > 0:
                        log_rets.append(math.log(p1 / p0))
                if len(log_rets) < 2:
                    continue
                mean = sum(log_rets) / len(log_rets)
                variance = sum((x - mean) ** 2 for x in log_rets) / (len(log_rets) - 1)
                ann_vol = round(math.sqrt(variance * 252) * 100, 4)
                results.append({"ticker": tk, "annualised_vol_pct": ann_vol})
            results.sort(key=lambda x: x["annualised_vol_pct"], reverse=True)
            return {
                "highest3": results[:3],
                "lowest3": results[-3:],
                "all_tickers": results,
                "meta": {"n_tickers": len(results)},
            }

        if metric == "max_volume":
            # Single largest daily volume record per ticker or overall
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            by_ticker: dict[str, list] = defaultdict(list)
            for r in filtered:
                by_ticker[r["ticker"]].append(r)
            results = []
            for tk, recs in by_ticker.items():
                best = max(recs, key=lambda r: r["volume"])
                results.append({"ticker": tk, "max_volume": int(best["volume"]), "date": str(best["date"])})
            results.sort(key=lambda x: -x["max_volume"])
            return {
                "best": results[0] if results else None,
                "top5": results[:5],
                "all_tickers": results,
            }

        if metric == "correlation":
            import math
            rows = _load_asx("all")  # always need all tickers for pairwise computation
            by_ticker: dict[str, list] = defaultdict(list)
            for r in _filter_asx(rows, date_from, date_to, ticker=None, exclude=excl):
                by_ticker[r["ticker"]].append(r)
            log_returns: dict[str, dict] = {}
            for tk, recs in by_ticker.items():
                recs.sort(key=lambda r: r["date"])
                lr = {}
                for i in range(1, len(recs)):
                    p0, p1 = recs[i-1]["close"], recs[i]["close"]
                    if p0 > 0 and p1 > 0:
                        lr[recs[i]["date"]] = math.log(p1 / p0)
                log_returns[tk] = lr
            tickers_list = sorted(log_returns.keys())
            if len(tickers_list) < 2:
                return {"error": "Need at least 2 tickers for correlation"}
            def pearson(xs, ys):
                n = len(xs)
                if n < 2:
                    return None
                mx, my = sum(xs)/n, sum(ys)/n
                num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
                dx = math.sqrt(sum((x-mx)**2 for x in xs))
                dy = math.sqrt(sum((y-my)**2 for y in ys))
                return round(num / (dx * dy), 4) if dx and dy else None
            pairs = []
            for i in range(len(tickers_list)):
                for j in range(i+1, len(tickers_list)):
                    tk1, tk2 = tickers_list[i], tickers_list[j]
                    common = set(log_returns[tk1]) & set(log_returns[tk2])
                    xs = [log_returns[tk1][d] for d in sorted(common)]
                    ys = [log_returns[tk2][d] for d in sorted(common)]
                    corr = pearson(xs, ys)
                    pairs.append({"pair": f"{tk1}-{tk2}", "correlation": corr, "n_days": len(xs)})
            pairs.sort(key=lambda x: x["correlation"] if x["correlation"] is not None else 0, reverse=True)
            return {"correlations": pairs, "meta": {"tickers": tickers_list, "n_pairs": len(pairs)}}

        if metric == "cross_year_extremes":
            # Best and worst annual return across ALL ticker-year combos (2015-2021)
            all_asx = _load_asx("all")
            combos = []
            for tk in all_tickers:
                if tk in excl:
                    continue
                for yr in range(2015, 2022):
                    ret = _annual_return(all_asx, tk, yr)
                    if ret is not None:
                        combos.append({"ticker": tk, "year": yr, "annual_return_pct": ret})
            combos.sort(key=lambda x: x["annual_return_pct"])
            return {
                "worst": combos[0] if combos else None,
                "best": combos[-1] if combos else None,
                "bottom5": combos[:5],
                "top5": list(reversed(combos[-5:])),
            }

        if metric == "count":
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            return {"result": len(filtered)}

        if metric == "list":
            rows = _load_asx(t or "all")
            filtered = _filter_asx(rows, date_from, date_to, ticker=t, exclude=excl)
            return {"records": [_asx_to_json(r) for r in filtered[:limit]],
                    "meta": {"total_filtered": len(filtered), "shown": min(limit, len(filtered))}}

        return {"error": f"Unknown metric '{metric}' for asx."}

    # ── AFR ──────────────────────────────────────────────────────────────────
    if dataset == "afr":
        if not os.path.isdir(AFR_DIR):
            return {"error": f"AFR data not found at {AFR_DIR}. Set DATA_DIR env var."}

        if not pattern and metric in ("share", "count", "count_by_month"):
            return {"error": "pattern= is required for AFR queries. Provide a regex string."}
        pat = pattern or ".*"
        try:
            rx = _re.compile(pat, _re.IGNORECASE)
        except _re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}

        if metric == "dataset_info":
            # Scan for year range without filtering
            years = set()
            total = 0
            for fn in sorted(glob.glob(os.path.join(AFR_DIR, "AFR_*.jsonl"))):
                base = os.path.basename(fn)
                try:
                    years.add(int(base[4:8]))
                except ValueError:
                    pass
                with open(fn) as f:
                    for line in f:
                        if line.strip():
                            total += 1
            return {
                "year_range": [min(years), max(years)] if years else [],
                "total_records": total,
                "note": "AFR news, records end Dec 2021",
            }

        if metric == "count":
            total = 0
            for r, _ in _iter_afr(date_from, date_to):
                if rx.search(_afr_text(r)):
                    total += 1
            return {"result": total, "pattern": pat,
                    "meta": {"date_from": date_from, "date_to": date_to}}

        if metric == "count_by_month":
            counts: dict[tuple, int] = defaultdict(int)
            total_matching = 0
            for r, rec_date in _iter_afr(date_from, date_to):
                if rx.search(_afr_text(r)):
                    counts[(rec_date.year, rec_date.month)] += 1
                    total_matching += 1
            ranked = sorted(counts.items(), key=lambda x: -x[1])
            records = [{"year": y, "month": m, "count": c} for (y, m), c in ranked[:limit]]
            # Also compute year totals
            year_totals: dict[int, int] = defaultdict(int)
            for (y, m), c in counts.items():
                year_totals[y] += c
            best_year = max(year_totals.items(), key=lambda x: x[1]) if year_totals else None
            return {
                "records": records,
                "best_month": records[0] if records else None,
                "best_year": {"year": best_year[0], "count": best_year[1]} if best_year else None,
                "total_matching": total_matching,
                "pattern": pat,
                "meta": {"date_from": date_from, "date_to": date_to},
            }

        if metric == "share":
            total_recs = 0
            matching = 0
            by_year: dict[int, dict] = {}
            for r, rec_date in _iter_afr(date_from, date_to):
                total_recs += 1
                yr = rec_date.year
                if yr not in by_year:
                    by_year[yr] = {"total": 0, "matching": 0}
                by_year[yr]["total"] += 1
                if rx.search(_afr_text(r)):
                    matching += 1
                    by_year[yr]["matching"] += 1
            share_pct = round(matching / total_recs * 100, 4) if total_recs else 0.0
            # Compute per-year shares
            year_shares = {}
            for yr, d in sorted(by_year.items()):
                year_shares[yr] = {
                    "matching": d["matching"],
                    "total": d["total"],
                    "share_pct": round(d["matching"] / d["total"] * 100, 4) if d["total"] else 0.0,
                }
            return {
                "result": share_pct,
                "matching": matching,
                "total": total_recs,
                "by_year": year_shares,
                "pattern": pat,
                "meta": {"date_from": date_from, "date_to": date_to},
            }

        return {"error": f"Unknown metric '{metric}' for afr. Options: count, count_by_month, share"}

    return {"error": f"Unknown dataset '{dataset}'. Use 'rba', 'asx', or 'afr'."}
