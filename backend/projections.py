#!/usr/bin/env python3
"""
fantasy-baseball-projections
projections.py — daily projection engine

Run manually:  python projections.py
Run via CI:    GitHub Actions triggers this every morning at 11:00 UTC (6am CDT)

Output:        docs/data.json  (read by index.html)
"""

import json
import os
import math
import datetime
import requests
from pybaseball import playerid_lookup, statcast_batter, batting_stats, pitching_stats
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# SECTION 1 — CONFIG & SCORING
# ─────────────────────────────────────────────

# Your league's exact point values per stat
SCORING = {
    "AB":    -0.5,
    "R":      0.2,
    "1B":     2.8,
    "2B":     3.8,
    "3B":     5.7,
    "HR":     7.5,
    "RBI":    0.2,
    "BB":     1.5,
    "K":     -0.2,
    "HBP":    1.5,
    "SAC":    0.5,
    "SB":     2.0,   # kept in config but not projected
    "CS":    -2.5,   # kept in config but not projected
    "GIDP":  -0.5,
}

# Projection model weights
WEIGHTS = {
    # How much recent 14-day form nudges the final score (3%)
    "recent_form": 0.03,

    # PA thresholds for career vs current season blending
    # At MIN_PA or below  → use career stats almost entirely  (career_floor)
    # At MAX_PA or above  → use current season almost entirely (season_ceil)
    # Between             → linear interpolation
    "pa_blend_min":      50,     # PA floor — lean heavily on career
    "pa_blend_max":     400,     # PA ceiling — lean heavily on current season
    "career_floor":      0.80,   # career weight at pa_blend_min  (80%)
    "season_ceil":       0.70,   # season weight at pa_blend_max  (70%)

    # Same logic for pitchers but using IP instead of PA
    "ip_blend_min":      20,     # IP floor
    "ip_blend_max":      80,     # IP ceiling
    "career_floor_p":    0.75,
    "season_ceil_p":     0.65,

    # Pitcher difficulty — how much each metric contributes to composite score
    # Must sum to 1.0
    "pitcher_xfip":      0.30,
    "pitcher_kpct":      0.20,
    "pitcher_bbpct":     0.15,
    "pitcher_hardpct":   0.15,
    "pitcher_whip":      0.10,
    "pitcher_stuffplus":  0.10,

    # BABIP — how aggressively to adjust H rate for lucky/unlucky BABIP
    # 0 = ignore BABIP difference, 1 = full correction
    "babip_correction":  0.40,

    # Max point adjustment from pitcher difficulty (up or down)
    "pitcher_adj_max":   3.5,

    # Max point adjustment from park factor
    "park_adj_max":      2.0,

    # Max point adjustment from recent form
    "form_adj_max":      1.5,
}

# Confidence score components and their weights (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    "pa_sample":        0.45,   # how many PA does the batter have this season
    "ip_sample":        0.25,   # how many IP does the pitcher have this season
    "babip_deviation":  0.30,   # how far is current BABIP from career BABIP
}

# PA/IP thresholds for confidence scoring
# Batter PA
CONF_PA_MIN  = 50    # 0% confidence contribution from PA sample
CONF_PA_MAX  = 400   # 100% confidence contribution from PA sample

# Pitcher IP
CONF_IP_MIN  = 10    # 0% confidence contribution from IP sample
CONF_IP_MAX  = 80    # 100% confidence contribution from IP sample

# BABIP deviation (absolute difference) thresholds
CONF_BABIP_GREAT = 0.020   # deviation <= this → full confidence from BABIP component
CONF_BABIP_BAD   = 0.100   # deviation >= this → zero confidence from BABIP component

# Current MLB season
CURRENT_SEASON = 2025

# Path to roster file (relative to this script)
ROSTER_PATH = "roster.json"

# Output path (GitHub Pages serves the docs/ folder)
OUTPUT_PATH = os.path.join("docs", "data.json")

# MLB Stats API base URL
MLB_API = "https://statsapi.mlb.com/api/v1"

# League-average values used as fallback when pitcher data is sparse/missing
LEAGUE_AVG = {
    "xfip":       4.20,
    "kpct":       0.222,
    "bbpct":      0.082,
    "hardpct":    0.365,
    "whip":       1.28,
    "stuffplus":  100.0,    # Stuff+ is normalized to 100 = league average
    "babip":      0.298,
    "hrate":      0.243,    # H per AB (roughly .243 BA)
    "hrrate":     0.034,    # HR per AB
    "xbhrate":    0.095,    # XBH per AB (2B + 3B)
    "bbpct_bat":  0.085,    # BB per PA (batters)
    "kpct_bat":   0.225,    # K per PA (batters)
    "abs_per_g":  3.7,      # expected AB per game
    "r_per_ab":   0.140,    # R per AB (league context)
    "rbi_per_ab": 0.135,    # RBI per AB (league context)
}


# ─────────────────────────────────────────────
# SECTION 2 — DATA FETCHING
# ─────────────────────────────────────────────
#
# Functions in this section:
#   load_roster()                → reads roster.json
#   get_today_matchups()         → MLB API: today's games + probable SPs
#   get_batter_season_stats()    → FanGraphs via pybaseball: current season
#   get_batter_career_stats()    → FanGraphs via pybaseball: multi-year career
#   get_batter_recent_stats()    → FanGraphs via pybaseball: last 14 days
#   get_pitcher_season_stats()   → FanGraphs via pybaseball: current season
#   get_pitcher_career_stats()   → FanGraphs via pybaseball: multi-year career
#   get_pitch_type_splits()      → Statcast via pybaseball: batter pitch type wOBA
#   get_pitcher_pitch_mix()      → Statcast via pybaseball: pitcher usage + whiff%
# ─────────────────────────────────────────────


def load_roster(path=ROSTER_PATH):
    """
    Reads roster.json and returns the list of player dicts.
    Filters out IL players — they get no projection.
    DTD players are kept but flagged so the dashboard can gray them out.
    """
    with open(path, "r") as f:
        data = json.load(f)
    players = data["players"]
    # Separate active/dtd from IL
    active = [p for p in players if not p["status"].startswith("il")]
    il     = [p for p in players if p["status"].startswith("il")]
    return active, il


def get_today_matchups():
    """
    Calls the MLB Stats API to get today's schedule.
    Returns a dict keyed by team abbreviation:
    {
      "NYY": {
        "opponent": "BAL",
        "venue": "Camden Yards",
        "venue_id": 2,
        "probable_pitcher": {
          "id": 669456,
          "name": "Corbin Burnes",
          "hand": "R"
        }
      },
      ...
    }
    If a probable pitcher isn't announced yet, probable_pitcher is None
    and the model falls back to league-average difficulty.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    url = f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=probablePitcher,venue,team"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARNING] MLB API schedule fetch failed: {e}")
        return {}

    matchups = {}

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            venue      = game.get("venue", {}).get("name", "Unknown")
            venue_id   = game.get("venue", {}).get("id", None)

            for side in ("home", "away"):
                team_data = game.get("teams", {}).get(side, {})
                team_abbr = team_data.get("team", {}).get("abbreviation", "")
                opp_side  = "away" if side == "home" else "home"
                opp_abbr  = game["teams"][opp_side]["team"].get("abbreviation", "")

                # Probable pitcher for the OPPOSING team (what this batter faces)
                opp_data  = game["teams"][opp_side]
                pitcher   = opp_data.get("probablePitcher", None)

                probable  = None
                if pitcher:
                    probable = {
                        "id":   pitcher.get("id"),
                        "name": pitcher.get("fullName", "Unknown"),
                        "hand": _get_pitcher_hand(pitcher.get("id"))
                    }

                matchups[team_abbr] = {
                    "opponent":         opp_abbr,
                    "venue":            venue,
                    "venue_id":         venue_id,
                    "probable_pitcher": probable,
                }

    return matchups


def _get_pitcher_hand(pitcher_id):
    """
    Helper: looks up a pitcher's throwing hand from the MLB people API.
    Returns "R", "L", or "R" as default if unavailable.
    """
    if not pitcher_id:
        return "R"
    try:
        url  = f"{MLB_API}/people/{pitcher_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        hand = data["people"][0]["pitchHand"]["code"]
        return hand  # "R" or "L"
    except Exception:
        return "R"  # safe default


def _mlb_stats_to_row(player_name, stat):
    """Converts an MLB Stats API stat dict to our internal batting row schema."""
    ab      = int(stat.get("atBats", 0))
    pa      = int(stat.get("plateAppearances", 0))
    h       = int(stat.get("hits", 0))
    doubles = int(stat.get("doubles", 0))
    triples = int(stat.get("triples", 0))
    hr      = int(stat.get("homeRuns", 0))
    bb      = int(stat.get("baseOnBalls", 0))
    k       = int(stat.get("strikeOuts", 0))
    hbp     = int(stat.get("hitByPitch", 0))
    babip   = float(stat.get("babip", LEAGUE_AVG["babip"]))

    ab_safe = max(ab, 1)
    pa_safe = max(pa, 1)

    return {
        "name":     player_name,
        "pa":       pa,
        "ab":       ab,
        "h":        h,
        "doubles":  doubles,
        "triples":  triples,
        "hr":       hr,
        "bb":       bb,
        "k":        k,
        "hbp":      hbp,
        "babip":    babip,
        "hrate":    h       / ab_safe,
        "hrrate":   hr      / ab_safe,
        "xbhrate":  (doubles + triples) / ab_safe,
        "bbpct":    bb      / pa_safe,
        "kpct":     k       / pa_safe,
        "hbprate":  hbp     / pa_safe,
    }


def _get_mlb_batting_stats_for_players(player_list, season=CURRENT_SEASON):
    """
    Fetches season batting stats for a list of players from the MLB Stats API.
    player_list: list of dicts with keys 'name' and 'mlb_id'
    Returns a dict keyed by normalised name: stat row dict.
    """
    results = {}
    for player in player_list:
        mlb_id = player.get("mlb_id")
        name   = player.get("name", "")
        if not mlb_id:
            continue
        try:
            url  = f"{MLB_API}/people/{mlb_id}/stats?stats=season&season={season}&group=hitting"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            row  = _mlb_stats_to_row(name, stat)
            norm = lookup_player_fg_name(mlb_id, name)
            results[norm] = row
        except Exception as e:
            pass  # silently fall through to league avg
    return results


def _get_mlb_career_batting_stats_for_players(player_list, start_year=2019):
    """
    Fetches career batting stats by aggregating multiple seasons from MLB Stats API.
    Returns a dict keyed by normalised name: aggregated stat row dict.
    """
    results = {}
    end_year = CURRENT_SEASON - 1

    for player in player_list:
        mlb_id = player.get("mlb_id")
        name   = player.get("name", "")
        if not mlb_id:
            continue

        totals = {"pa":0,"ab":0,"h":0,"doubles":0,"triples":0,
                  "hr":0,"bb":0,"k":0,"hbp":0,"babip_sum":0,"seasons":0}

        for yr in range(start_year, end_year + 1):
            try:
                url  = f"{MLB_API}/people/{mlb_id}/stats?stats=season&season={yr}&group=hitting"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                splits = data.get("stats", [{}])[0].get("splits", [])
                if not splits:
                    continue
                stat = splits[0].get("stat", {})
                totals["pa"]       += int(stat.get("plateAppearances", 0))
                totals["ab"]       += int(stat.get("atBats",           0))
                totals["h"]        += int(stat.get("hits",             0))
                totals["doubles"]  += int(stat.get("doubles",          0))
                totals["triples"]  += int(stat.get("triples",          0))
                totals["hr"]       += int(stat.get("homeRuns",         0))
                totals["bb"]       += int(stat.get("baseOnBalls",      0))
                totals["k"]        += int(stat.get("strikeOuts",       0))
                totals["hbp"]      += int(stat.get("hitByPitch",       0))
                babip = float(stat.get("babip", LEAGUE_AVG["babip"]))
                totals["babip_sum"] += babip
                totals["seasons"]   += 1
            except Exception:
                pass

        if totals["ab"] == 0:
            continue

        ab_safe = max(totals["ab"], 1)
        pa_safe = max(totals["pa"], 1)
        babip   = totals["babip_sum"] / max(totals["seasons"], 1)

        row = {
            "name":     name,
            "pa":       totals["pa"],
            "ab":       totals["ab"],
            "h":        totals["h"],
            "doubles":  totals["doubles"],
            "triples":  totals["triples"],
            "hr":       totals["hr"],
            "bb":       totals["bb"],
            "k":        totals["k"],
            "hbp":      totals["hbp"],
            "babip":    babip,
            "hrate":    totals["h"]                               / ab_safe,
            "hrrate":   totals["hr"]                              / ab_safe,
            "xbhrate":  (totals["doubles"] + totals["triples"])   / ab_safe,
            "bbpct":    totals["bb"]                              / pa_safe,
            "kpct":     totals["k"]                               / pa_safe,
            "hbprate":  totals["hbp"]                             / pa_safe,
        }
        norm = lookup_player_fg_name(mlb_id, name)
        results[norm] = row

    return results


def get_batter_season_stats(season=CURRENT_SEASON, player_list=None):
    """
    Pulls current-season batting stats from the MLB Stats API.
    player_list: list of dicts with 'name' and 'mlb_id' keys.
    Returns a dict keyed by normalised name (not a DataFrame).
    """
    if not player_list:
        return {}
    print(f"  Fetching MLB API season batting stats for {len(player_list)} players...")
    return _get_mlb_batting_stats_for_players(player_list, season)


def get_batter_career_stats(start_year=2019, end_year=None, player_list=None):
    """
    Pulls career batting stats from the MLB Stats API (2019–last season).
    player_list: list of dicts with 'name' and 'mlb_id' keys.
    Returns a dict keyed by normalised name.
    """
    if not player_list:
        return {}
    print(f"  Fetching MLB API career batting stats for {len(player_list)} players...")
    return _get_mlb_career_batting_stats_for_players(player_list, start_year)


def get_batter_recent_stats(mlb_id, days=14):
    """
    Pulls last N days of Statcast data for a single batter.
    Used for the recent form adjustment (3% weight).
    Returns a dict of rate stats, or None if insufficient data.
    """
    end_dt   = datetime.date.today()
    start_dt = end_dt - datetime.timedelta(days=days)

    try:
        df = statcast_batter(
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            player_id=mlb_id
        )
        if df is None or len(df) < 5:
            return None  # too few events to trust

        # Count plate appearance outcomes
        pa    = len(df[df["events"].notna()])
        hits  = len(df[df["events"].isin(["single","double","triple","home_run"])])
        hrs   = len(df[df["events"] == "home_run"])
        bbs   = len(df[df["events"] == "walk"])
        ks    = len(df[df["events"] == "strikeout"])
        abs_  = pa - bbs - len(df[df["events"] == "hit_by_pitch"]) \
                   - len(df[df["events"].isin(["sac_fly","sac_bunt"])])
        abs_  = max(abs_, 1)

        return {
            "pa":      pa,
            "hrate":   hits / abs_,
            "hrrate":  hrs  / abs_,
            "bbpct":   bbs  / max(pa, 1),
            "kpct":    ks   / max(pa, 1),
        }
    except Exception as e:
        print(f"  [WARNING] Statcast recent stats failed for id {mlb_id}: {e}")
        return None


def get_batter_pitch_type_splits(mlb_id, season=CURRENT_SEASON):
    """
    Pulls Statcast data for a batter for the current season and aggregates
    wOBA by pitch type bucket (fastball / breaking ball / offspeed).

    Pitch type groupings:
      Fastball:     FF, FT, SI, FC (4-seam, 2-seam, sinker, cutter)
      Breaking ball: SL, CU, KC, SV, ST (slider, curve, knuckle-curve, slurve, sweeper)
      Offspeed:     CH, FS, SC (changeup, splitter, screwball)

    Returns a dict:
    {
      "fastball":     {"woba": 0.340, "pa": 210},
      "breaking":     {"woba": 0.198, "pa": 180},
      "offspeed":     {"woba": 0.280, "pa":  95},
    }
    Only includes buckets with pa >= 75 (our minimum threshold).
    Returns empty dict if data unavailable.
    """
    FASTBALL    = {"FF", "FT", "SI", "FC"}
    BREAKING    = {"SL", "CU", "KC", "SV", "ST"}
    OFFSPEED    = {"CH", "FS", "SC"}
    MIN_PA      = 75

    start = f"{season}-01-01"
    end   = datetime.date.today().strftime("%Y-%m-%d")

    try:
        df = statcast_batter(start, end, player_id=mlb_id)
        if df is None or df.empty:
            return {}

        # Only rows where there's a pitch type and an outcome
        df = df[df["pitch_type"].notna() & df["events"].notna()].copy()

        results = {}
        for bucket, types in [
            ("fastball", FASTBALL),
            ("breaking", BREAKING),
            ("offspeed", OFFSPEED),
        ]:
            subset = df[df["pitch_type"].isin(types)]
            pa = len(subset)
            if pa < MIN_PA:
                continue  # not enough data — silently skip this bucket
            # Statcast has estimated_woba_using_speedangle; fall back to woba_value
            if "estimated_woba_using_speedangle" in subset.columns:
                woba = subset["estimated_woba_using_speedangle"].mean()
            elif "woba_value" in subset.columns:
                woba = subset["woba_value"].mean()
            else:
                continue
            if pd.isna(woba):
                continue
            results[bucket] = {"woba": round(float(woba), 3), "pa": int(pa)}

        return results

    except Exception as e:
        print(f"  [WARNING] Statcast pitch splits failed for id {mlb_id}: {e}")
        return {}


def _get_mlb_pitching_stats_for_pitcher(pitcher_id, pitcher_name, season=CURRENT_SEASON):
    """
    Fetches season pitching stats for a single pitcher from the MLB Stats API.
    Returns a dict with our internal pitching schema, or None if not found.
    """
    try:
        url  = f"{MLB_API}/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat = splits[0].get("stat", {})

        ip  = float(stat.get("inningsPitched", 0))
        era = float(stat.get("era",  LEAGUE_AVG["xfip"]))
        whip= float(stat.get("whip", LEAGUE_AVG["whip"]))
        bf  = int(stat.get("battersFaced",   0))
        k   = int(stat.get("strikeOuts",     0))
        bb  = int(stat.get("baseOnBalls",    0))

        bf_safe = max(bf, 1)
        return {
            "name":      pitcher_name,
            "ip":        ip,
            "era":       era,
            "xfip":      era,       # use ERA as xFIP proxy — no xFIP in MLB API
            "whip":      whip,
            "kpct":      k  / bf_safe,
            "bbpct":     bb / bf_safe,
            "hardpct":   LEAGUE_AVG["hardpct"],   # not in MLB API
            "stuffplus":  LEAGUE_AVG["stuffplus"], # not in MLB API
            "source":    "mlb_api",
        }
    except Exception:
        return None


def _get_mlb_pitching_career_for_pitcher(pitcher_id, pitcher_name, start_year=2019):
    """Aggregates career pitching stats from MLB Stats API."""
    end_year = CURRENT_SEASON - 1
    totals = {"ip":0.0,"era_sum":0.0,"whip_sum":0.0,"bf":0,"k":0,"bb":0,"seasons":0}

    for yr in range(start_year, end_year + 1):
        try:
            url  = f"{MLB_API}/people/{pitcher_id}/stats?stats=season&season={yr}&group=pitching"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            totals["ip"]       += float(stat.get("inningsPitched", 0))
            totals["era_sum"]  += float(stat.get("era",  LEAGUE_AVG["xfip"]))
            totals["whip_sum"] += float(stat.get("whip", LEAGUE_AVG["whip"]))
            totals["bf"]       += int(stat.get("battersFaced",  0))
            totals["k"]        += int(stat.get("strikeOuts",    0))
            totals["bb"]       += int(stat.get("baseOnBalls",   0))
            totals["seasons"]  += 1
        except Exception:
            pass

    if totals["ip"] == 0:
        return None

    bf_safe = max(totals["bf"], 1)
    seasons = max(totals["seasons"], 1)
    return {
        "name":      pitcher_name,
        "ip":        totals["ip"],
        "era":       totals["era_sum"]  / seasons,
        "xfip":      totals["era_sum"]  / seasons,
        "whip":      totals["whip_sum"] / seasons,
        "kpct":      totals["k"]  / bf_safe,
        "bbpct":     totals["bb"] / bf_safe,
        "hardpct":   LEAGUE_AVG["hardpct"],
        "stuffplus":  LEAGUE_AVG["stuffplus"],
        "source":    "mlb_api_career",
    }


def get_pitcher_season_stats(season=CURRENT_SEASON, pitcher_ids=None):
    """
    Pulls season pitching stats from MLB Stats API for a list of pitcher IDs.
    pitcher_ids: list of (id, name) tuples.
    Returns a dict keyed by normalised name.
    """
    if not pitcher_ids:
        return {}
    results = {}
    for pid, pname in pitcher_ids:
        row = _get_mlb_pitching_stats_for_pitcher(pid, pname, season)
        if row:
            results[lookup_player_fg_name(pid, pname)] = row
    return results


def get_pitcher_career_stats(start_year=2019, end_year=None, pitcher_ids=None):
    """
    Pulls career pitching stats from MLB Stats API for a list of pitcher IDs.
    pitcher_ids: list of (id, name) tuples.
    Returns a dict keyed by normalised name.
    """
    if not pitcher_ids:
        return {}
    results = {}
    for pid, pname in pitcher_ids:
        row = _get_mlb_pitching_career_for_pitcher(pid, pname, start_year)
        if row:
            results[lookup_player_fg_name(pid, pname)] = row
    return results


def get_pitcher_pitch_mix(pitcher_mlb_id, season=CURRENT_SEASON):
    """
    Pulls Statcast pitch-level data for a pitcher and computes
    per-bucket usage % and whiff rate.

    Minimum 50 pitches per bucket required.

    Returns:
    {
      "fastball":  {"usage": 0.42, "whiff_pct": 0.24, "woba_against": 0.310, "pitches": 580},
      "breaking":  {"usage": 0.38, "whiff_pct": 0.34, "woba_against": 0.201, "pitches": 520},
      "offspeed":  {"usage": 0.20, "whiff_pct": 0.38, "woba_against": 0.188, "pitches": 275},
    }
    Only buckets with >= 50 pitches are included.
    Returns empty dict if data unavailable.
    """
    from pybaseball import statcast_pitcher

    FASTBALL  = {"FF", "FT", "SI", "FC"}
    BREAKING  = {"SL", "CU", "KC", "SV", "ST"}
    OFFSPEED  = {"CH", "FS", "SC"}
    MIN_PITCH = 50

    start = f"{season}-01-01"
    end   = datetime.date.today().strftime("%Y-%m-%d")

    try:
        df = statcast_pitcher(start, end, player_id=pitcher_mlb_id)
        if df is None or df.empty:
            return {}

        df = df[df["pitch_type"].notna()].copy()
        total_pitches = len(df)
        if total_pitches == 0:
            return {}

        # Whiff = swing and miss
        df["whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked"])

        results = {}
        for bucket, types in [
            ("fastball", FASTBALL),
            ("breaking", BREAKING),
            ("offspeed", OFFSPEED),
        ]:
            subset = df[df["pitch_type"].isin(types)]
            n = len(subset)
            if n < MIN_PITCH:
                continue

            usage     = n / total_pitches
            whiff_pct = subset["whiff"].sum() / max(n, 1)

            # wOBA against on contact (events only)
            events = subset[subset["events"].notna()]
            if "woba_value" in events.columns and len(events) > 0:
                woba_against = float(events["woba_value"].mean())
            else:
                woba_against = LEAGUE_AVG["xfip"] / 10  # rough fallback

            results[bucket] = {
                "usage":         round(float(usage), 3),
                "whiff_pct":     round(float(whiff_pct), 3),
                "woba_against":  round(float(woba_against), 3),
                "pitches":       int(n),
            }

        return results

    except Exception as e:
        print(f"  [WARNING] Statcast pitch mix failed for pitcher id {pitcher_mlb_id}: {e}")
        return {}


def lookup_player_fg_name(mlb_id, mlb_name):
    """
    FanGraphs and MLB Stats API use different player IDs.
    We match on name since pybaseball returns name-keyed DataFrames.
    Returns the best matching name string to use as a lookup key.
    This is a fuzzy match — handles minor name differences.
    """
    # Simple normalisation: lowercase, strip accents, remove Jr/Sr/II etc
    import unicodedata
    def norm(s):
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower().strip()
        for suffix in [" jr.", " sr.", " ii", " iii", " iv"]:
            s = s.replace(suffix, "")
        return s
    return norm(mlb_name)


def _get_mlb_platoon_splits_for_player(mlb_id, player_name, start_year=2022):
    """
    Fetches vs-Left and vs-Right batting splits from MLB Stats API.
    Aggregates across start_year to current season for stability.
    Returns dict: {"vL": stat_row, "vR": stat_row} or {} if unavailable.
    Each stat_row has same schema as _mlb_stats_to_row output.
    """
    end_year = datetime.date.today().year
    totals = {
        "vL": {"pa":0,"ab":0,"h":0,"doubles":0,"triples":0,"hr":0,"bb":0,"k":0,"hbp":0},
        "vR": {"pa":0,"ab":0,"h":0,"doubles":0,"triples":0,"hr":0,"bb":0,"k":0,"hbp":0},
    }

    for yr in range(start_year, end_year + 1):
        try:
            url  = f"{MLB_API}/people/{mlb_id}/stats?stats=statSplits&season={yr}&group=hitting&sitCodes=vl,vr"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            for s in splits:
                desc = s.get("split", {}).get("description", "")
                stat = s.get("stat", {})
                key  = "vL" if "Left" in desc else "vR" if "Right" in desc else None
                if not key:
                    continue
                totals[key]["pa"]      += int(stat.get("plateAppearances", 0))
                totals[key]["ab"]      += int(stat.get("atBats",           0))
                totals[key]["h"]       += int(stat.get("hits",             0))
                totals[key]["doubles"] += int(stat.get("doubles",          0))
                totals[key]["triples"] += int(stat.get("triples",          0))
                totals[key]["hr"]      += int(stat.get("homeRuns",         0))
                totals[key]["bb"]      += int(stat.get("baseOnBalls",      0))
                totals[key]["k"]       += int(stat.get("strikeOuts",       0))
                totals[key]["hbp"]     += int(stat.get("hitByPitch",       0))
        except Exception:
            pass

    result = {}
    for key, t in totals.items():
        if t["ab"] < 20:
            continue   # too little data — skip this split
        ab_s = max(t["ab"], 1)
        pa_s = max(t["pa"], 1)
        result[key] = {
            "pa":      t["pa"],
            "ab":      t["ab"],
            "hrate":   t["h"]                           / ab_s,
            "hrrate":  t["hr"]                          / ab_s,
            "xbhrate": (t["doubles"] + t["triples"])    / ab_s,
            "bbpct":   t["bb"]                          / pa_s,
            "kpct":    t["k"]                           / pa_s,
            "hbprate": t["hbp"]                         / pa_s,
        }
    return result


def get_all_platoon_splits(player_list):
    """
    Fetches platoon splits for all active players.
    Returns dict keyed by mlb_id: {"vL": row, "vR": row}
    """
    result = {}
    for player in player_list:
        mlb_id = player.get("mlb_id")
        name   = player.get("name", "")
        if not mlb_id:
            continue
        splits = _get_mlb_platoon_splits_for_player(mlb_id, name)
        if splits:
            result[mlb_id] = splits
    return result


def fetch_all_data(active_players, matchups):
    """
    Master fetch function. Now uses MLB Stats API for batting/pitching stats.
    Returns dicts keyed by normalised name instead of DataFrames.
    """
    batter_list = [
        {"name": p["name"], "mlb_id": p["mlb_id"]}
        for p in active_players
        if p["status"] != "dtd" and not p["status"].startswith("il")
    ]

    seen = set()
    pitcher_ids = []
    for team, matchup in matchups.items():
        sp = matchup.get("probable_pitcher")
        if sp and sp["id"] and sp["id"] not in seen:
            pitcher_ids.append((sp["id"], sp["name"]))
            seen.add(sp["id"])

    print("  Fetching MLB API season batting stats...")
    batter_season = get_batter_season_stats(player_list=batter_list)
    print(f"    Got {len(batter_season)} players")

    print("  Fetching MLB API career batting stats...")
    batter_career = get_batter_career_stats(player_list=batter_list)
    print(f"    Got {len(batter_career)} players")

    print("  Fetching MLB API season pitching stats...")
    pitcher_season = get_pitcher_season_stats(pitcher_ids=pitcher_ids)
    print(f"    Got {len(pitcher_season)} pitchers")

    print("  Fetching MLB API career pitching stats...")
    pitcher_career = get_pitcher_career_stats(pitcher_ids=pitcher_ids)
    print(f"    Got {len(pitcher_career)} pitchers")

    pitch_splits  = {}
    pitcher_mixes = {}

    print("  Fetching MLB API platoon splits (vs L / vs R)...")
    platoon_splits = get_all_platoon_splits(batter_list)
    print(f"    Got splits for {len(platoon_splits)} players")

    print("  Fetching Statcast pitch type splits for batters...")
    for player in active_players:
        mlb_id = player["mlb_id"]
        if player["status"] == "dtd":
            continue
        print(f"    {player['name']}...")
        pitch_splits[mlb_id] = get_batter_pitch_type_splits(mlb_id)

    print("  Fetching Statcast pitch mix for probable pitchers...")
    seen_pitchers = set()
    for team, matchup in matchups.items():
        sp = matchup.get("probable_pitcher")
        if sp and sp["id"] and sp["id"] not in seen_pitchers:
            print(f"    {sp['name']}...")
            pitcher_mixes[sp["id"]] = get_pitcher_pitch_mix(sp["id"])
            seen_pitchers.add(sp["id"])

    return batter_season, batter_career, pitcher_season, pitcher_career, \
           pitch_splits, pitcher_mixes, platoon_splits


def blend_weight(n, n_min, n_max, floor, ceil):
    """
    Returns the weight to give the CURRENT SEASON stats (0.0 → 1.0).
    The complement (1 - result) is the weight given to CAREER stats.

    At n <= n_min  → season weight = (1 - floor)   e.g. 0.20 at 50 PA
    At n >= n_max  → season weight = ceil           e.g. 0.70 at 400 PA
    Between        → linear interpolation

    Example with batter defaults (floor=0.80, ceil=0.70):
      50  PA → season=0.20, career=0.80
      200 PA → season=0.43, career=0.57
      400 PA → season=0.70, career=0.30
      600 PA → season=0.70, career=0.30  (capped at ceil)
    """
    if n <= n_min:
        return 1.0 - floor
    if n >= n_max:
        return ceil
    # Linear interpolation
    t = (n - n_min) / (n_max - n_min)
    season_at_min = 1.0 - floor
    return season_at_min + t * (ceil - season_at_min)


def _safe_get(row, col, fallback):
    """
    Safely retrieves a value from a DataFrame row (or dict).
    Returns fallback if missing, null, or NaN.
    """
    if row is None:
        return fallback
    try:
        val = row[col] if isinstance(row, dict) else row.get(col, fallback)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return fallback
        return float(val)
    except Exception:
        return fallback


def blend_batter_stats(season_row, career_row, season_pa):
    """
    Blends current season and career batting stats into one stat dict.

    season_row  — row from batter_season DataFrame (or None if not found)
    career_row  — row from batter_career DataFrame (or None if not found)
    season_pa   — PA count this season (drives the blend weight)

    Returns a dict with blended rate stats ready for projection:
    {
      "pa":        int,    current season PA (for confidence scoring)
      "hrate":     float,  projected H per AB
      "hrrate":    float,  projected HR per AB
      "xbhrate":   float,  projected XBH (2B+3B) per AB
      "bbpct":     float,  projected BB per PA
      "kpct":      float,  projected K per PA
      "hbprate":   float,  projected HBP per PA
      "babip_curr": float, current season BABIP
      "babip_car":  float, career BABIP
      "abs_per_g":  float, expected AB per game (league avg for now)
      "source":    str,    "season" / "career" / "blended" / "league_avg"
    }
    """
    w = blend_weight(
        season_pa,
        WEIGHTS["pa_blend_min"],
        WEIGHTS["pa_blend_max"],
        WEIGHTS["career_floor"],
        WEIGHTS["season_ceil"],
    )
    career_w = 1.0 - w

    def blend(col, fallback):
        s = _safe_get(season_row, col, None)
        c = _safe_get(career_row, col, None)
        if s is not None and c is not None:
            return w * s + career_w * c
        if s is not None:
            return s
        if c is not None:
            return c
        return fallback

    # Determine data source label for transparency
    if season_row is not None and career_row is not None:
        source = "blended"
    elif season_row is not None:
        source = "season"
    elif career_row is not None:
        source = "career"
    else:
        source = "league_avg"

    return {
        "pa":         int(season_pa),
        "hrate":      blend("hrate",   LEAGUE_AVG["hrate"]),
        "hrrate":     blend("hrrate",  LEAGUE_AVG["hrrate"]),
        "xbhrate":    blend("xbhrate", LEAGUE_AVG["xbhrate"]),
        "bbpct":      blend("bbpct",   LEAGUE_AVG["bbpct_bat"]),
        "kpct":       blend("kpct",    LEAGUE_AVG["kpct_bat"]),
        "hbprate":    blend("hbprate", 0.010),
        "babip_curr": _safe_get(season_row, "babip", LEAGUE_AVG["babip"]),
        "babip_car":  _safe_get(career_row, "babip", LEAGUE_AVG["babip"]),
        "abs_per_g":  LEAGUE_AVG["abs_per_g"],
        "source":     source,
        "bats":       "R",  # overwritten by get_batter_stats_for_player
    }


def blend_pitcher_stats(season_row, career_row, season_ip):
    """
    Blends current season and career pitching stats.
    Stuff+ always uses the career/multi-year value (more stable by design).

    Returns:
    {
      "ip":        float,  current season IP (for confidence scoring)
      "xfip":      float,
      "kpct":      float,
      "bbpct":     float,
      "hardpct":   float,
      "whip":      float,
      "stuffplus":  float,  always multi-year average
      "source":    str,
    }
    """
    w = blend_weight(
        season_ip,
        WEIGHTS["ip_blend_min"],
        WEIGHTS["ip_blend_max"],
        WEIGHTS["career_floor_p"],
        WEIGHTS["season_ceil_p"],
    )
    career_w = 1.0 - w

    def blend(col, fallback):
        s = _safe_get(season_row, col, None)
        c = _safe_get(career_row, col, None)
        if s is not None and c is not None:
            return w * s + career_w * c
        if s is not None:
            return s
        if c is not None:
            return c
        return fallback

    if season_row is not None and career_row is not None:
        source = "blended"
    elif season_row is not None:
        source = "season"
    elif career_row is not None:
        source = "career"
    else:
        source = "league_avg"

    # Stuff+ — always career average, never current season alone
    stuffplus = _safe_get(career_row, "stuffplus", LEAGUE_AVG["stuffplus"])
    if stuffplus == LEAGUE_AVG["stuffplus"]:
        # Try season as secondary fallback
        stuffplus = _safe_get(season_row, "stuffplus", LEAGUE_AVG["stuffplus"])

    return {
        "ip":        float(season_ip),
        "xfip":      blend("xfip",     LEAGUE_AVG["xfip"]),
        "kpct":      blend("kpct",     LEAGUE_AVG["kpct"]),
        "bbpct":     blend("bbpct",    LEAGUE_AVG["bbpct"]),
        "hardpct":   blend("hardpct",  LEAGUE_AVG["hardpct"]),
        "whip":      blend("whip",     LEAGUE_AVG["whip"]),
        "stuffplus":  stuffplus,
        "source":    source,
    }


def get_batter_stats_for_player(name, mlb_id, batter_season, batter_career):
    """
    Looks up a player in the season and career dicts (keyed by norm name),
    then blends them. batter_season and batter_career are now dicts not DataFrames.
    """
    norm_name  = lookup_player_fg_name(mlb_id, name)
    season_row = batter_season.get(norm_name) if isinstance(batter_season, dict) else None
    career_row = batter_career.get(norm_name) if isinstance(batter_career, dict) else None
    season_pa  = int(season_row.get("pa", 0)) if season_row else 0

    if season_row is None and career_row is None:
        print(f"    [WARNING] No batting stats found for {name} — using league avg")

    result = blend_batter_stats(season_row, career_row, season_pa)
    return result


def get_pitcher_stats_for_pitcher(name, pitcher_id, pitcher_season, pitcher_career):
    """
    Looks up a pitcher in the season and career dicts (keyed by norm name),
    then blends them.
    """
    if not name:
        return blend_pitcher_stats(None, None, 0)

    norm_name  = lookup_player_fg_name(pitcher_id or 0, name)
    season_row = pitcher_season.get(norm_name) if isinstance(pitcher_season, dict) else None
    career_row = pitcher_career.get(norm_name) if isinstance(pitcher_career, dict) else None
    season_ip  = float(season_row.get("ip", 0.0)) if season_row else 0.0

    if season_row is None and career_row is None:
        print(f"    [WARNING] No pitching stats found for {name} — using league avg")

    return blend_pitcher_stats(season_row, career_row, season_ip)


def blend_recent_form(base_stats, recent_stats):
    """
    Nudges the blended base stats using recent 14-day Statcast data.
    Weight is WEIGHTS["recent_form"] = 3%.

    recent_stats is the dict from get_batter_recent_stats(), or None.
    Returns a new stat dict with the recent form adjustment applied,
    plus a "form_delta" key showing the raw point impact for the dashboard.
    """
    if recent_stats is None or recent_stats.get("pa", 0) < 10:
        # Not enough recent data — return base stats unchanged
        return {**base_stats, "form_delta": 0.0}

    w_form = WEIGHTS["recent_form"]
    w_base = 1.0 - w_form

    blended = {**base_stats}  # copy
    for stat in ["hrate", "hrrate", "bbpct", "kpct"]:
        if stat in recent_stats:
            blended[stat] = w_base * base_stats[stat] + w_form * recent_stats[stat]

    # Compute approximate point delta from form adjustment (for dashboard display)
    # We'll compute the real delta in Section 6, but store a rough estimate here
    h_delta  = (blended["hrate"]  - base_stats["hrate"])  * base_stats["abs_per_g"]
    hr_delta = (blended["hrrate"] - base_stats["hrrate"]) * base_stats["abs_per_g"]
    bb_delta = (blended["bbpct"]  - base_stats["bbpct"])  * (base_stats["abs_per_g"] + base_stats["bbpct"] * base_stats["abs_per_g"])

    form_delta = (
        h_delta  * SCORING["1B"] +   # rough — treats all hits as singles for estimate
        hr_delta * SCORING["HR"] +
        bb_delta * SCORING["BB"]
    )
    # Cap per the config max
    form_delta = max(-WEIGHTS["form_adj_max"], min(WEIGHTS["form_adj_max"], form_delta))

    blended["form_delta"] = round(form_delta, 2)
    return blended



# ─────────────────────────────────────────────
# SECTION 4 — BABIP ADJUSTMENT
# ─────────────────────────────────────────────
#
# BABIP (Batting Average on Balls In Play) is one of the best signals
# for luck vs. true talent. A hitter with a career .310 BABIP running
# .240 this season is likely getting unlucky — their balls in play are
# finding gloves at an unusual rate. We nudge their projected H rate
# upward to partially correct for this.
#
# We apply 40% of the implied correction (WEIGHTS["babip_correction"]).
# Full correction would be too aggressive — BABIP takes time to normalize
# and there are legitimate reasons for short-term deviation (injury,
# lineup spot, batted ball profile changes).
#
# The deviation also feeds the confidence score — a player whose BABIP
# is far from career norms has more variance in their projection.
#
# Functions:
#   babip_adjust(batter_stats)  → adjusted stat dict + adjustment metadata
# ─────────────────────────────────────────────


def babip_adjust(batter_stats):
    """
    Adjusts the projected H rate based on BABIP deviation from career.

    How BABIP connects to H rate:
      BABIP = (H - HR) / (AB - K - HR + SF)
      Rearranging: H ≈ BABIP * (AB - K - HR) + HR
      So H rate ≈ BABIP * (1 - kpct - hrrate) + hrrate

    The adjustment:
      implied_hrate  = what H rate SHOULD be given career BABIP
      current_hrate  = what H rate IS given current BABIP
      delta          = implied - current  (positive = unlucky, negative = lucky)
      correction     = delta * WEIGHTS["babip_correction"]  (40% of full delta)
      adjusted_hrate = current_hrate + correction

    Returns the input dict with:
      - hrate updated to the adjusted value
      - babip_adj metadata added for the confidence scorer and dashboard
    """
    babip_curr = batter_stats["babip_curr"]
    babip_car  = batter_stats["babip_car"]
    kpct       = batter_stats["kpct"]
    hrrate     = batter_stats["hrrate"]
    hrate      = batter_stats["hrate"]

    # How much of each AB results in a ball in play
    # (AB - K - HR) / AB  ≈  1 - kpct - hrrate
    bip_rate = max(1.0 - kpct - hrrate, 0.01)  # guard against negative

    # Implied H rate if the player were hitting to career BABIP
    implied_hrate  = babip_car  * bip_rate + hrrate
    current_hrate  = babip_curr * bip_rate + hrrate

    # Raw delta — how far off current is from career-implied
    delta = implied_hrate - current_hrate

    # Apply partial correction
    correction = delta * WEIGHTS["babip_correction"]
    adjusted_hrate = hrate + correction

    # Clamp — never let the adjustment move H rate more than 20% in either direction
    max_move = hrate * 0.20
    adjusted_hrate = max(hrate - max_move, min(hrate + max_move, adjusted_hrate))
    adjusted_hrate = max(adjusted_hrate, 0.050)  # floor — always project some hits

    # BABIP deviation magnitude (used by confidence scorer)
    babip_deviation = abs(babip_curr - babip_car)

    # Direction label for dashboard display
    if correction > 0.005:
        direction = "unlucky — H rate nudged up"
    elif correction < -0.005:
        direction = "lucky — H rate nudged down"
    else:
        direction = "in line with career"

    result = {**batter_stats}
    result["hrate"] = adjusted_hrate
    result["babip_adj"] = {
        "babip_curr":    round(babip_curr, 3),
        "babip_car":     round(babip_car,  3),
        "deviation":     round(babip_deviation, 3),
        "correction":    round(correction, 4),
        "direction":     direction,
    }
    return result


def babip_confidence_penalty(babip_adj_meta):
    """
    Converts BABIP deviation into a 0.0–1.0 confidence score component.
    Large deviation = low confidence (high variance projection).
    Small deviation = high confidence (stable projection).

    0.0 = worst confidence (deviation >= CONF_BABIP_BAD  = 0.100)
    1.0 = best confidence  (deviation <= CONF_BABIP_GREAT = 0.020)
    """
    deviation = babip_adj_meta.get("deviation", 0.0)
    if deviation <= CONF_BABIP_GREAT:
        return 1.0
    if deviation >= CONF_BABIP_BAD:
        return 0.0
    # Linear interpolation between thresholds
    t = (deviation - CONF_BABIP_GREAT) / (CONF_BABIP_BAD - CONF_BABIP_GREAT)
    return round(1.0 - t, 3)



# ─────────────────────────────────────────────
# SECTION 5 — PITCHER DIFFICULTY SCORE
# ─────────────────────────────────────────────
#
# Converts blended pitcher stats into two outputs:
#
#   1. pitcher_difficulty_adj  — a point adjustment (roughly -3.5 to +3.5)
#      applied to the batter's projected fantasy score.
#      Negative = elite pitcher suppresses the batter.
#      Positive = weak pitcher boosts the batter.
#
#   2. pitch_type_adj — an additional adjustment (capped ±1.0) based on
#      how the specific batter matches up against this pitcher's
#      pitch mix (fastball / breaking ball / offspeed buckets).
#
# Functions:
#   pitcher_difficulty_adj(pitcher_stats)
#   pitch_type_matchup_adj(batter_splits, pitcher_mix)
#   format_pitch_rows(pitch_adj_detail)   → dashboard-ready list
# ─────────────────────────────────────────────


# League-average z-score reference ranges for each pitching metric.
# Used to normalize each metric to a -1.0 → +1.0 scale before weighting.
# Values represent one standard deviation from league average.
# Direction is flipped where higher = worse for batters (xFIP, WHIP, BB%)
# and where higher = better for batters (K%, Hard%).

PITCHER_NORMS = {
    # (league_avg, one_std_dev, higher_is_harder_for_batter)
    "xfip":      (4.20, 0.55, False),  # lower xFIP = harder pitcher
    "kpct":      (0.222, 0.045, True), # higher K% = harder pitcher
    "bbpct":     (0.082, 0.025, False), # lower BB% = harder pitcher (fewer free passes)
    "hardpct":   (0.365, 0.045, False), # lower Hard% = harder pitcher
    "whip":      (1.28,  0.15, False),  # lower WHIP = harder pitcher
    "stuffplus":  (100.0, 12.0, True),  # higher Stuff+ = harder pitcher
}


def _normalize_pitcher_metric(value, metric):
    """
    Normalizes a single pitcher metric to a -1.0 → +1.0 scale
    relative to league average, capped at ±2 standard deviations.

    Returns a positive value when the pitcher is HARDER than average
    (i.e. bad for the batter), negative when EASIER than average.
    """
    avg, std, higher_is_harder = PITCHER_NORMS[metric]
    z = (value - avg) / std
    z = max(-2.0, min(2.0, z))  # cap at ±2 std devs
    # Flip sign so positive always means "harder for batter"
    return z if higher_is_harder else -z


def pitcher_difficulty_adj(pitcher_stats, pitcher_name="Unknown", xfip_display=None):
    """
    Produces a single point adjustment for pitcher difficulty.

    Steps:
      1. Normalize each metric to -1.0 → +1.0 (positive = harder for batter)
      2. Weighted sum using WEIGHTS["pitcher_*"] coefficients
      3. Scale to fantasy point impact using WEIGHTS["pitcher_adj_max"]
      4. Negate — a harder pitcher = negative adjustment for the batter

    Returns a dict:
    {
      "adj":          float,   point adjustment (negative = hard pitcher)
      "composite":    float,   raw weighted score before scaling (-1 to +1)
      "display_xfip": float,   xFIP to show on dashboard
      "label":        str,     e.g. "Burnes · xFIP 3.21"
      "components":   dict,    per-metric normalized scores (for debugging)
    }
    """
    metrics = ["xfip", "kpct", "bbpct", "hardpct", "whip", "stuffplus"]
    weight_keys = {
        "xfip":      "pitcher_xfip",
        "kpct":      "pitcher_kpct",
        "bbpct":     "pitcher_bbpct",
        "hardpct":   "pitcher_hardpct",
        "whip":      "pitcher_whip",
        "stuffplus":  "pitcher_stuffplus",
    }

    composite  = 0.0
    components = {}

    for metric in metrics:
        val  = pitcher_stats.get(metric, LEAGUE_AVG.get(metric, 0))
        norm = _normalize_pitcher_metric(val, metric)
        w    = WEIGHTS[weight_keys[metric]]
        composite  += norm * w
        components[metric] = round(norm, 3)

    # Scale composite (-1 to +1) to point adjustment
    # Positive composite = harder pitcher = negative adj for batter
    adj = -(composite * WEIGHTS["pitcher_adj_max"])
    adj = max(-WEIGHTS["pitcher_adj_max"], min(WEIGHTS["pitcher_adj_max"], adj))

    xfip_val = xfip_display or pitcher_stats.get("xfip", LEAGUE_AVG["xfip"])

    return {
        "adj":          round(adj, 2),
        "composite":    round(composite, 3),
        "display_xfip": round(xfip_val, 2),
        "label":        f"{pitcher_name} · xFIP {xfip_val:.2f}",
        "source":       pitcher_stats.get("source", "unknown"),
        "components":   components,
    }


def pitch_type_matchup_adj(batter_splits, pitcher_mix):
    """
    Computes the pitch type matchup adjustment.

    batter_splits — dict from get_batter_pitch_type_splits()
      e.g. {"fastball": {"woba": 0.401, "pa": 210}, "breaking": {...}}

    pitcher_mix   — dict from get_pitcher_pitch_mix()
      e.g. {"fastball": {"usage": 0.42, "whiff_pct": 0.24, ...}, ...}

    Logic per bucket:
      batter_score  = (batter_woba - LEAGUE_AVG_WOBA) / STD_WOBA
                      negative = vulnerability, positive = strength
      pitcher_score = (pitcher_whiff - LEAGUE_AVG_WHIFF) / STD_WHIFF
                      + (LEAGUE_AVG_WOBA_AGAINST - pitcher_woba_against) / STD_WOBA
                      positive = elite pitch
      bucket_impact = batter_score * pitcher_score * usage_weight * SCALE

    Total capped at ±1.0 pts.

    Returns:
    {
      "total_adj": float,   total point adjustment (capped ±1.0)
      "detail": [           list of rows for dashboard display
        {
          "bucket":      "breaking",
          "label":       "Breaking ball",
          "badge_class": "pb-break",
          "usage_pct":   0.38,
          "whiff_pct":   0.34,
          "batter_woba": 0.198,
          "direction":   "vulnerability",
          "adj":         -0.8,
          "pitcher_name_short": "Burnes",
        },
        ...
      ]
    }
    Only includes buckets where both batter and pitcher have sufficient data.
    Only includes buckets where abs(adj) >= 0.15 (meaningful threshold).
    """

    # League average reference values for normalisation
    LEAGUE_WOBA       = 0.315
    STD_WOBA          = 0.055
    LEAGUE_WHIFF      = 0.245
    STD_WHIFF         = 0.060
    MIN_IMPACT        = 0.15   # don't show rows below this threshold
    SCALE             = 1.8    # maps normalised product to point range
    CAP               = 1.0    # total adjustment cap

    BUCKET_META = {
        "fastball":  {"label": "Fastball",     "badge_class": "pb-fast"},
        "breaking":  {"label": "Breaking ball","badge_class": "pb-break"},
        "offspeed":  {"label": "Offspeed",     "badge_class": "pb-off"},
    }

    total_adj = 0.0
    detail    = []

    for bucket in ["fastball", "breaking", "offspeed"]:
        b_data = batter_splits.get(bucket)
        p_data = pitcher_mix.get(bucket)

        # Skip if either side lacks sufficient sample
        if not b_data or not p_data:
            continue

        batter_woba     = b_data["woba"]
        usage           = p_data["usage"]
        whiff_pct       = p_data["whiff_pct"]
        woba_against    = p_data["woba_against"]

        # Normalise batter vulnerability
        # Negative z = batter struggles (woba below league avg)
        batter_z = (batter_woba - LEAGUE_WOBA) / STD_WOBA

        # Normalise pitcher quality on this pitch type
        # Positive = elite (high whiff, low woba against)
        whiff_z  = (whiff_pct  - LEAGUE_WHIFF) / STD_WHIFF
        woba_z   = (LEAGUE_WOBA - woba_against) / STD_WOBA  # flipped: lower woba = harder
        pitcher_z = (whiff_z + woba_z) / 2.0

        # Bucket impact:
        # batter_z negative + pitcher_z positive = bad for batter (negative adj)
        # batter_z positive + pitcher_z negative = good for batter (positive adj)
        impact = batter_z * pitcher_z * usage * SCALE

        total_adj += impact

        if abs(impact) < MIN_IMPACT:
            continue  # not meaningful enough to show on dashboard

        if batter_woba < LEAGUE_WOBA - 0.020:
            direction = "vulnerability"
        elif batter_woba > LEAGUE_WOBA + 0.020:
            direction = "strength"
        else:
            direction = "neutral"

        meta = BUCKET_META[bucket]
        detail.append({
            "bucket":       bucket,
            "label":        meta["label"],
            "badge_class":  meta["badge_class"],
            "usage_pct":    round(usage, 3),
            "whiff_pct":    round(whiff_pct, 3),
            "batter_woba":  round(batter_woba, 3),
            "direction":    direction,
            "adj":          round(impact, 2),
        })

    # Sort detail rows by abs impact descending (biggest effect first)
    detail.sort(key=lambda x: abs(x["adj"]), reverse=True)

    # Cap total
    total_adj = max(-CAP, min(CAP, total_adj))

    return {
        "total_adj": round(total_adj, 2),
        "detail":    detail,
    }


def format_pitch_rows(pitch_adj_detail, pitcher_name):
    """
    Formats the pitch type matchup detail list into dashboard-ready strings.

    Returns a list of dicts ready to be written into data.json:
    [
      {
        "label":       "Breaking ball",
        "badge_class": "pb-break",
        "meta":        "Burnes 38% usage · 34% whiff",
        "detail":      "Judge .198 wOBA vs breaking — vulnerability",
        "adj":         -0.8,
        "adj_str":     "−0.8",
        "adj_class":   "fneg",
      },
      ...
    ]
    """
    rows = []
    sp_short = pitcher_name.split()[-1] if pitcher_name else "SP"

    for item in pitch_adj_detail:
        adj     = item["adj"]
        adj_str = f"+{adj:.1f}" if adj >= 0 else f"−{abs(adj):.1f}"
        cls     = "fpos" if adj > 0 else ("fneg" if adj < 0 else "fneu")

        rows.append({
            "label":       item["label"],
            "badge_class": item["badge_class"],
            "meta":        f"{sp_short} {item['usage_pct']*100:.0f}% usage · {item['whiff_pct']*100:.0f}% whiff",
            "detail":      f"{item['batter_woba']:.3f} wOBA vs {item['label'].lower()} — {item['direction']}",
            "adj":         adj,
            "adj_str":     adj_str,
            "adj_class":   cls,
        })

    return rows



# ─────────────────────────────────────────────
# SECTION 6 — PARK FACTORS
# ─────────────────────────────────────────────
#
# Park factors adjust projected R/RBI and HR rates based on the
# characteristics of today's ballpark.
#
# Two factors per park:
#   run_factor  — multiplier on R and RBI projection
#                 1.05 = 5% more runs than average
#                 0.95 = 5% fewer runs than average
#   hr_factor   — multiplier on HR projection specifically
#                 Parks like Coors and Cincinnati boost HR.
#                 Parks like Oakland and Petco suppress them.
#
# Source: 3-year averaged park factors from FanGraphs (2022-2024).
# Updated manually each offseason or when a park changes dimensions.
#
# MLB venue IDs are from the MLB Stats API (used to match today's games).
#
# Functions:
#   get_park_factors(venue_id, venue_name)  → {"run": float, "hr": float, "name": str}
#   park_factor_adj(batter_stats, park)     → point adjustment float
# ─────────────────────────────────────────────


# Park factor table keyed by MLB venue ID.
# Venue IDs verified from MLB Stats API 2026 season.
# run_factor and hr_factor are multipliers around 1.0 (league average).
# Source: FanGraphs park factors 2022-2024 averages.
PARK_FACTORS = {
    # AL East
    3313: {"name": "Yankee Stadium",         "run": 1.03, "hr": 1.11},
    2:    {"name": "Camden Yards",            "run": 1.05, "hr": 1.08},
    3:    {"name": "Fenway Park",             "run": 1.04, "hr": 0.93},
    14:   {"name": "Rogers Centre",           "run": 1.02, "hr": 1.05},
    12:   {"name": "Tropicana Field",         "run": 0.96, "hr": 0.94},

    # AL Central
    4:    {"name": "Rate Field",              "run": 1.05, "hr": 1.10},
    5:    {"name": "Progressive Field",       "run": 0.96, "hr": 0.91},
    2394: {"name": "Comerica Park",           "run": 0.97, "hr": 0.93},
    7:    {"name": "Kauffman Stadium",        "run": 0.97, "hr": 0.95},
    3312: {"name": "Target Field",            "run": 0.97, "hr": 0.94},

    # AL West
    1:    {"name": "Angel Stadium",           "run": 0.97, "hr": 0.97},
    2392: {"name": "Daikin Park",             "run": 1.01, "hr": 0.99},
    2529: {"name": "Sutter Health Park",      "run": 0.94, "hr": 0.87},
    680:  {"name": "T-Mobile Park",           "run": 0.97, "hr": 0.93},
    5325: {"name": "Globe Life Field",        "run": 1.02, "hr": 1.05},

    # NL East
    4705: {"name": "Truist Park",             "run": 1.01, "hr": 1.04},
    3289: {"name": "Citi Field",              "run": 0.95, "hr": 0.89},
    2681: {"name": "Citizens Bank Park",      "run": 1.06, "hr": 1.12},
    3309: {"name": "Nationals Park",          "run": 0.99, "hr": 1.00},
    4169: {"name": "loanDepot park",          "run": 0.96, "hr": 0.93},

    # NL Central
    17:   {"name": "Wrigley Field",           "run": 1.03, "hr": 1.04},
    2602: {"name": "Great American Ball Park","run": 1.06, "hr": 1.14},
    32:   {"name": "American Family Field",   "run": 1.03, "hr": 1.06},
    31:   {"name": "PNC Park",                "run": 0.97, "hr": 0.94},
    2889: {"name": "Busch Stadium",           "run": 0.97, "hr": 0.96},

    # NL West
    15:   {"name": "Chase Field",             "run": 1.04, "hr": 1.08},
    19:   {"name": "Coors Field",             "run": 1.16, "hr": 1.22},
    22:   {"name": "Dodger Stadium",          "run": 0.98, "hr": 0.97},
    2680: {"name": "Petco Park",              "run": 0.94, "hr": 0.88},
    2395: {"name": "Oracle Park",             "run": 0.94, "hr": 0.87},
}

# Fallback for unknown venues
PARK_AVERAGE = {"name": "Unknown Park", "run": 1.00, "hr": 1.00}

# Name-based lookup for when venue_id doesn't match
# (MLB API sometimes returns different IDs for same park)
PARK_NAME_LOOKUP = {
    "yankee stadium":          4,
    "camden yards":            2,
    "fenway park":             3313,
    "rogers centre":           3289,
    "tropicana field":         3901,
    "guaranteed rate field":   5,
    "progressive field":       7,
    "comerica park":           2394,
    "kauffman stadium":        7,
    "target field":            3309,
    "angel stadium":           1,
    "minute maid park":        11,
    "oakland coliseum":        10,
    "t-mobile park":           680,
    "globe life field":        5,
    "truist park":             3289,
    "citi field":              3410,
    "citizens bank park":      2392,
    "nationals park":          3091,
    "loandepot park":          4169,
    "wrigley field":           17,
    "great american ball park":2602,
    "american family field":   31,
    "pnc park":                3312,
    "busch stadium":           2889,
    "chase field":             22,
    "coors field":             16,
    "dodger stadium":          22,
    "petco park":              2680,
    "oracle park":             2395,
}


def get_park_factors(venue_id=None, venue_name=None):
    """
    Returns park factor dict for a given venue.
    Tries venue_id first, falls back to name matching, then league average.

    Returns: {"name": str, "run": float, "hr": float}
    """
    # Try by ID first
    if venue_id and venue_id in PARK_FACTORS:
        return PARK_FACTORS[venue_id]

    # Try by name (case-insensitive)
    if venue_name:
        norm = venue_name.lower().strip()
        vid  = PARK_NAME_LOOKUP.get(norm)
        if vid and vid in PARK_FACTORS:
            return PARK_FACTORS[vid]
        # Partial match fallback
        for key, vid in PARK_NAME_LOOKUP.items():
            if key in norm or norm in key:
                if vid in PARK_FACTORS:
                    return PARK_FACTORS[vid]

    return PARK_AVERAGE


def park_factor_adj(batter_stats, park, pitcher_difficulty=None):
    """
    Converts park factors into a fantasy point adjustment.

    How it works:
      HR contribution  = hrrate * abs_per_g * SCORING["HR"]
      R  contribution  = r_per_ab * abs_per_g * SCORING["R"]
      RBI contribution = rbi_per_ab * abs_per_g * SCORING["RBI"]

      HR adj  = HR contribution  * (hr_factor  - 1.0)
      Run adj = Run contribution * (run_factor - 1.0)  (covers R + RBI)

      total_adj = HR adj + Run adj

    The adjustment is relative to league average (1.0 = no adjustment).
    Coors Field (hr=1.22) gives a big positive boost.
    Petco Park (hr=0.87) gives a meaningful negative.

    Returns a dict:
    {
      "adj":         float,   total point adjustment
      "hr_adj":      float,   HR component
      "run_adj":     float,   R+RBI component
      "park_name":   str,
      "run_factor":  float,
      "hr_factor":   float,
    }
    """
    run_factor = park["run"]
    hr_factor  = park["hr"]

    hrrate     = batter_stats["hrrate"]
    abs_per_g  = batter_stats["abs_per_g"]

    # Expected fantasy points from HR per game at league-average park
    hr_pts_base  = hrrate * abs_per_g * SCORING["HR"]
    # Expected fantasy points from R and RBI per game
    run_pts_base = (
        LEAGUE_AVG["r_per_ab"]   * abs_per_g * SCORING["R"] +
        LEAGUE_AVG["rbi_per_ab"] * abs_per_g * SCORING["RBI"]
    )

    hr_adj  = hr_pts_base  * (hr_factor  - 1.0)
    run_adj = run_pts_base * (run_factor - 1.0)

    total_adj = hr_adj + run_adj
    total_adj = max(-WEIGHTS["park_adj_max"], min(WEIGHTS["park_adj_max"], total_adj))

    return {
        "adj":        round(total_adj, 2),
        "hr_adj":     round(hr_adj,    2),
        "run_adj":    round(run_adj,   2),
        "park_name":  park["name"],
        "run_factor": run_factor,
        "hr_factor":  hr_factor,
    }



# ─────────────────────────────────────────────
# SECTION 7 — FINAL PROJECTION + CONFIDENCE
# ─────────────────────────────────────────────
#
# This section assembles all prior sections into a single projection
# per player. It also computes the confidence score.
#
# Functions:
#   compute_confidence(batter_stats, pitcher_stats, babip_meta)
#   project_stat_line(batter_stats)
#   score_stat_line(stat_line)
#   build_factor_breakdown(base_pts, platoon_adj, pitcher_diff,
#                          park_adj, form_delta, pitch_type)
#   project_player(player, matchup, all_data)
#   project_all(active_players, il_players, matchups, all_data)
# ─────────────────────────────────────────────


def compute_confidence(batter_stats, pitcher_stats, babip_meta):
    """
    Produces a 0–100 confidence score for the projection.

    Three components weighted by CONFIDENCE_WEIGHTS:
      pa_sample    (45%) — how many PA does the batter have this season
      ip_sample    (25%) — how many IP does the pitcher have this season
      babip_dev    (30%) — how far is current BABIP from career

    Each component is scored 0.0–1.0 then combined.

    Returns:
    {
      "score":       int,    0-100
      "label":       str,    "high" / "medium" / "low"
      "pa_score":    float,
      "ip_score":    float,
      "babip_score": float,
    }
    """
    # PA component
    pa = batter_stats.get("pa", 0)
    pa_score = min(1.0, max(0.0,
        (pa - CONF_PA_MIN) / max(CONF_PA_MAX - CONF_PA_MIN, 1)
    ))

    # IP component
    ip = pitcher_stats.get("ip", 0)
    ip_score = min(1.0, max(0.0,
        (ip - CONF_IP_MIN) / max(CONF_IP_MAX - CONF_IP_MIN, 1)
    ))

    # BABIP component
    babip_score = babip_confidence_penalty(babip_meta)

    # Weighted composite
    raw = (
        pa_score    * CONFIDENCE_WEIGHTS["pa_sample"] +
        ip_score    * CONFIDENCE_WEIGHTS["ip_sample"] +
        babip_score * CONFIDENCE_WEIGHTS["babip_deviation"]
    )

    score = int(round(raw * 100))
    score = max(5, min(95, score))  # floor/ceiling — never claim 0% or 100%

    if score >= 60:
        label = "high"
    elif score >= 38:
        label = "medium"
    else:
        label = "low"

    return {
        "score":       score,
        "label":       label,
        "pa_score":    round(pa_score,    3),
        "ip_score":    round(ip_score,    3),
        "babip_score": round(babip_score, 3),
    }


def project_stat_line(batter_stats):
    """
    Converts blended rate stats into a projected per-game stat line.

    Expected stats per game:
      AB   = abs_per_g
      H    = hrate * AB
      HR   = hrrate * AB
      XBH  = xbhrate * AB   (2B + 3B combined)
      1B   = H - HR - XBH
      BB   = bbpct * PA      (PA ≈ AB + BB + HBP)
      K    = kpct * PA
      HBP  = hbprate * PA
      R    = r_per_ab * AB  (league-avg run environment)
      RBI  = rbi_per_ab * AB

    Returns a dict of projected counting stats per game.
    """
    ab      = batter_stats["abs_per_g"]
    hrate   = batter_stats["hrate"]
    hrrate  = batter_stats["hrrate"]
    xbhrate = batter_stats["xbhrate"]
    bbpct   = batter_stats["bbpct"]
    kpct    = batter_stats["kpct"]
    hbprate = batter_stats["hbprate"]

    h   = hrate   * ab
    hr  = hrrate  * ab
    xbh = xbhrate * ab
    xbh = min(xbh, h - hr)   # XBH can't exceed non-HR hits
    s   = max(h - hr - xbh, 0)

    # Approximate PA from AB + BB + HBP
    # BB and HBP are PA-based rates, so we need to solve for PA:
    # PA = AB + BB + HBP  →  PA = AB / (1 - bbpct - hbprate)
    denom = max(1.0 - bbpct - hbprate, 0.70)
    pa    = ab / denom

    bb  = bbpct   * pa
    k   = kpct    * pa
    hbp = hbprate * pa

    r   = LEAGUE_AVG["r_per_ab"]   * ab
    rbi = LEAGUE_AVG["rbi_per_ab"] * ab

    # XBH split: rough 80/20 split between doubles and triples
    doubles = xbh * 0.82
    triples = xbh * 0.18

    return {
        "ab":      round(ab,      2),
        "h":       round(h,       2),
        "singles": round(s,       2),
        "doubles": round(doubles, 2),
        "triples": round(triples, 2),
        "hr":      round(hr,      2),
        "bb":      round(bb,      2),
        "k":       round(k,       2),
        "hbp":     round(hbp,     3),
        "r":       round(r,       2),
        "rbi":     round(rbi,     2),
        "pa":      round(pa,      2),
    }


def score_stat_line(stat_line):
    """
    Converts a projected stat line to fantasy points using the
    exact league scoring from SCORING config.

    Returns:
    {
      "total":      float,   total projected fantasy points
      "components": dict,    per-stat point contribution
    }
    """
    components = {
        "AB":  stat_line["ab"]      * SCORING["AB"],
        "R":   stat_line["r"]       * SCORING["R"],
        "1B":  stat_line["singles"] * SCORING["1B"],
        "2B":  stat_line["doubles"] * SCORING["2B"],
        "3B":  stat_line["triples"] * SCORING["3B"],
        "HR":  stat_line["hr"]      * SCORING["HR"],
        "RBI": stat_line["rbi"]     * SCORING["RBI"],
        "BB":  stat_line["bb"]      * SCORING["BB"],
        "K":   stat_line["k"]       * SCORING["K"],
        "HBP": stat_line["hbp"]     * SCORING["HBP"],
    }
    total = sum(components.values())

    return {
        "total":      round(total, 2),
        "components": {k: round(v, 2) for k, v in components.items()},
    }


def build_factor_breakdown(base_pts, platoon_adj, pitcher_diff,
                           park_adj, form_delta, pitch_type_adj,
                           babip_adj_meta):
    """
    Assembles the factor breakdown shown in the expanded card.
    Each factor is a labelled row with a numeric adjustment.

    Returns a list of dicts:
    [
      {"label": "Base rate (season)", "adj": 12.4, "adj_str": "+12.4", "cls": "fneu"},
      {"label": "Platoon (R vs RHP)", "adj": 1.8,  "adj_str": "+1.8",  "cls": "fpos"},
      ...
    ]
    """
    def fmt(val, neutral_range=0.05):
        s   = f"+{val:.1f}" if val >= 0 else f"\u2212{abs(val):.1f}"
        cls = "fneu" if abs(val) < neutral_range else ("fpos" if val > 0 else "fneg")
        return s, cls

    rows = []

    # Base rate — always neutral class (it's the starting point, not an adjustment)
    rows.append({
        "label":   "Base rate (season)",
        "adj":     round(base_pts, 2),
        "adj_str": f"+{base_pts:.1f}",
        "cls":     "fneu",
        "mult":    "",
        "pill":    "baseline",
    })

    # Platoon
    p_str, p_cls = fmt(platoon_adj)
    rows.append({
        "label":   pitcher_diff.get("platoon_label", "Platoon split"),
        "adj":     round(platoon_adj, 2),
        "adj_str": p_str,
        "cls":     p_cls,
        "mult":    "",
        "pill":    "boosted" if platoon_adj > 0.05 else "suppressed" if platoon_adj < -0.05 else "neutral",
    })

    # Pitcher difficulty
    pd_str, pd_cls = fmt(pitcher_diff["adj"])
    h_mult  = pitcher_diff["components"].get("xfip", 0)
    k_mult  = pitcher_diff["components"].get("kpct", 0)
    pd_mult = f"H ×{1 - h_mult*0.1:.2f} · K ×{1 + k_mult*0.1:.2f}"
    rows.append({
        "label":   f"Pitcher ({pitcher_diff['label']})",
        "adj":     pitcher_diff["adj"],
        "adj_str": pd_str,
        "cls":     pd_cls,
        "mult":    pd_mult,
        "pill":    "suppressed" if pitcher_diff["adj"] < -0.05 else "boosted" if pitcher_diff["adj"] > 0.05 else "neutral",
    })

    # Park factor
    pk_str, pk_cls = fmt(park_adj["adj"])
    pk_mult = f"HR ×{park_adj['hr_factor']:.2f} · R ×{park_adj['run_factor']:.2f}"
    park_label = park_adj["park_name"]
    rows.append({
        "label":   f"Park factor ({park_label})",
        "adj":     park_adj["adj"],
        "adj_str": pk_str,
        "cls":     pk_cls,
        "mult":    pk_mult,
        "pill":    "boosted" if park_adj["adj"] > 0.05 else "suppressed" if park_adj["adj"] < -0.05 else "neutral",
    })

    # BABIP adjustment (only show if meaningful)
    if babip_adj_meta and abs(babip_adj_meta.get("correction", 0)) > 0.002:
        direction = babip_adj_meta.get("direction", "")
        babip_pts = babip_adj_meta["correction"] * SCORING["1B"] * 3.7
        b_str, b_cls = fmt(babip_pts)
        b_mult = f"H ×{1 + babip_adj_meta['correction']*2:.2f}"
        b_pill = "unlucky" if babip_pts > 0 else "lucky"
        rows.append({
            "label":   f"BABIP ({babip_adj_meta['babip_curr']:.3f} curr · {babip_adj_meta['babip_car']:.3f} career)",
            "adj":     round(babip_pts, 2),
            "adj_str": b_str,
            "cls":     b_cls,
            "mult":    b_mult,
            "pill":    b_pill,
        })

    # Recent form
    f_str, f_cls = fmt(form_delta)
    rows.append({
        "label":   "Recent form (14d)",
        "adj":     round(form_delta, 2),
        "adj_str": f_str,
        "cls":     f_cls,
        "mult":    f"H ×{1 + form_delta*0.02:.2f}",
        "pill":    "slight boost" if form_delta > 0.05 else "slight drag" if form_delta < -0.05 else "neutral",
    })

    return rows


def project_player(player, matchup, batter_season, batter_career,
                   pitcher_season, pitcher_career,
                   pitch_splits, pitcher_mixes, platoon_splits=None):
    """
    Full projection pipeline for a single player.

    Returns a projection dict ready for data.json:
    {
      "name":          str,
      "mlb_id":        int,
      "positions":     list,
      "status":        str,
      "team":          str,
      "opponent":      str,
      "venue":         str,
      "probable_sp":   str,
      "sp_hand":       str,
      "proj_pts":      float,
      "confidence":    {score, label},
      "stat_line":     {ab, h, hr, bb, k, r, rbi},
      "factors":       list of factor rows,
      "pitch_rows":    list of pitch type rows,
      "babip_note":    str,
      "data_source":   str,
    }
    DTD players return a skeleton with proj_pts=None.
    """
    name   = player["name"]
    mlb_id = player["mlb_id"]
    status = player["status"]

    # DTD — return skeleton, no projection
    if status == "dtd":
        return {
            "name":        name,
            "mlb_id":      mlb_id,
            "positions":   player["positions"],
            "status":      "dtd",
            "proj_pts":    None,
            "confidence":  None,
            "stat_line":   None,
            "factors":     [],
            "pitch_rows":  [],
        }

    # --- Matchup info ---
    team     = _infer_team(player)
    sp       = matchup.get("probable_pitcher") if matchup else None
    sp_name  = sp["name"]  if sp else "TBD"
    sp_id    = sp["id"]    if sp else None
    sp_hand  = sp["hand"]  if sp else "R"
    venue    = matchup.get("venue", "Unknown")     if matchup else "Unknown"
    venue_id = matchup.get("venue_id")             if matchup else None
    opponent = matchup.get("opponent", "???")      if matchup else "???"

    # --- Step 1: Batter stats (blended) ---
    batter_stats = get_batter_stats_for_player(
        name, mlb_id, batter_season, batter_career
    )

    # --- Step 2: BABIP adjustment ---
    batter_stats = babip_adjust(batter_stats)
    babip_meta   = batter_stats.get("babip_adj", {})

    # --- Step 3: Recent form blend ---
    recent = get_batter_recent_stats(mlb_id)
    batter_stats = blend_recent_form(batter_stats, recent)
    form_delta   = batter_stats.get("form_delta", 0.0)

    # --- Step 4: Platoon adjustment ---
    batter_stats["bats"] = player.get("bats", "R")
    platoon_result = _platoon_adjustment(
        batter_stats, sp_hand, batter_season, batter_career, name, mlb_id,
        platoon_splits=platoon_splits
    )

    # Handle both return cases:
    # Real splits: returns (split_row, label)
    # Fallback:    returns (None, label, delta_woba)
    if len(platoon_result) == 3:
        # League average fallback
        _, platoon_label, platoon_woba_delta = platoon_result
        platoon_split_row = None
    else:
        platoon_split_row, platoon_label = platoon_result
        platoon_woba_delta = 0.0

    batter_stats["platoon_label"] = platoon_label

    # If we have real split rates, override base batter stats with split rates
    # This is the key improvement — use actual vs-L or vs-R rates directly
    if platoon_split_row:
        for rate in ["hrate", "hrrate", "xbhrate", "bbpct", "kpct", "hbprate"]:
            if rate in platoon_split_row:
                batter_stats[rate] = platoon_split_row[rate]

    # --- Step 5: Pitcher difficulty ---
    pitcher_stats = get_pitcher_stats_for_pitcher(
        sp_name, sp_id, pitcher_season, pitcher_career
    )
    pitch_diff = pitcher_difficulty_adj(pitcher_stats, sp_name)

    # --- Step 6: Park factor ---
    park  = get_park_factors(venue_id, venue)
    p_adj = park_factor_adj(batter_stats, park)

    # --- Step 7: Apply all factors as rate multipliers ---
    # Convert each adjustment into a rate multiplier and apply to
    # the batter's stat rates BEFORE scoring. This means adjustments
    # compound through the scoring math rather than stacking additively.

    adjusted = dict(batter_stats)  # copy

    # Platoon — if real splits were applied in Step 4, rates are already
    # set correctly. If fallback, apply legacy wOBA delta as multiplier.
    if platoon_split_row:
        # Real splits already applied to batter_stats in Step 4
        # adjusted already copied from batter_stats so no additional change needed
        platoon_h_mult  = 1.0
        platoon_hr_mult = 1.0
    else:
        # Fallback: league average wOBA delta as rate multiplier
        platoon_h_mult  = 1.0 + platoon_woba_delta * 1.5
        platoon_hr_mult = 1.0 + platoon_woba_delta * 0.8
        adjusted["hrate"]  = max(adjusted["hrate"]  * platoon_h_mult,  0.050)
        adjusted["hrrate"] = max(adjusted["hrrate"] * platoon_hr_mult, 0.005)

    # Pitcher difficulty — affects hrate, kpct, bbpct
    # Use normalized component scores from pitch_diff
    comps     = pitch_diff.get("components", {})
    xfip_z    = comps.get("xfip",   0.0)   # positive = harder pitcher
    kpct_z    = comps.get("kpct",   0.0)
    bbpct_z   = comps.get("bbpct",  0.0)

    # Harder pitcher suppresses hits, elevates Ks, lowers BBs
    pitcher_h_mult  = 1.0 - xfip_z  * 0.06   # max ~12% suppression at 2 std devs
    pitcher_hr_mult = 1.0 - xfip_z  * 0.05
    pitcher_k_mult  = 1.0 + kpct_z  * 0.08
    pitcher_bb_mult = 1.0 - bbpct_z * 0.05

    adjusted["hrate"]  = max(adjusted["hrate"]  * pitcher_h_mult,  0.050)
    adjusted["hrrate"] = max(adjusted["hrrate"] * pitcher_hr_mult, 0.005)
    adjusted["kpct"]   = max(adjusted["kpct"]   * pitcher_k_mult,  0.050)
    adjusted["bbpct"]  = max(adjusted["bbpct"]  * pitcher_bb_mult, 0.020)

    # Park factor — affects hrrate and run environment
    hr_mult  = park["hr"]   # e.g. 1.11 for Camden Yards
    run_mult = park["run"]  # e.g. 1.05
    adjusted["hrrate"]    = max(adjusted["hrrate"] * hr_mult,  0.005)
    # Run environment affects r_per_ab and rbi_per_ab implicitly via scoring
    # Store run_mult for use in project_stat_line
    adjusted["run_mult"]  = run_mult

    # Recent form is already blended into the rates by blend_recent_form()
    # form_delta is kept for display only — no additional rate adjustment needed

    # --- Step 8: Score the adjusted stat line ---
    stat_line  = project_stat_line(adjusted)
    scored     = score_stat_line(stat_line)
    proj_pts_base = scored["total"]

    # Apply run environment multiplier to R and RBI components
    r_rbi_base = (
        stat_line["r"]   * SCORING["R"] +
        stat_line["rbi"] * SCORING["RBI"]
    )
    r_rbi_adj  = r_rbi_base * (run_mult - 1.0)
    proj_pts_base += r_rbi_adj

    # Compute base_pts from UNADJUSTED stat line for dashboard display
    base_stat_line = project_stat_line(batter_stats)
    base_score     = score_stat_line(base_stat_line)
    base_pts       = base_score["total"]

    # Compute individual factor point impacts for dashboard
    # (difference between adjusted and base for each factor)
    def _pts_with(rates):
        sl = project_stat_line(rates)
        return score_stat_line(sl)["total"]

    # Platoon impact
    platoon_only = dict(batter_stats)
    platoon_only["hrate"]  = max(batter_stats["hrate"]  * platoon_h_mult,  0.050)
    platoon_only["hrrate"] = max(batter_stats["hrrate"] * platoon_hr_mult, 0.005)
    platoon_impact = round(_pts_with(platoon_only) - base_pts, 2)

    # Pitcher impact
    pitcher_only = dict(batter_stats)
    pitcher_only["hrate"]  = max(batter_stats["hrate"]  * pitcher_h_mult,  0.050)
    pitcher_only["hrrate"] = max(batter_stats["hrrate"] * pitcher_hr_mult, 0.005)
    pitcher_only["kpct"]   = max(batter_stats["kpct"]   * pitcher_k_mult,  0.050)
    pitcher_only["bbpct"]  = max(batter_stats["bbpct"]  * pitcher_bb_mult, 0.020)
    pitcher_impact = round(_pts_with(pitcher_only) - base_pts, 2)

    # Park impact
    park_only = dict(batter_stats)
    park_only["hrrate"]   = max(batter_stats["hrrate"] * hr_mult, 0.005)
    park_only["run_mult"] = run_mult
    park_sl    = project_stat_line(park_only)
    park_scored = score_stat_line(park_sl)["total"]
    park_r_rbi  = (park_sl["r"] * SCORING["R"] + park_sl["rbi"] * SCORING["RBI"]) * (run_mult - 1.0)
    park_impact = round((park_scored + park_r_rbi) - base_pts, 2)

    # Override p_adj with computed impact for dashboard display
    p_adj["adj"] = park_impact

    # --- Step 9: Pitch type matchup (still additive — small wOBA-based adj) ---
    b_splits  = pitch_splits.get(mlb_id, {})
    p_mix     = pitcher_mixes.get(sp_id, {}) if sp_id else {}
    pt_result = pitch_type_matchup_adj(b_splits, p_mix)
    pt_rows   = format_pitch_rows(pt_result["detail"], sp_name)

    # --- Step 10: Confidence ---
    confidence = compute_confidence(adjusted, pitcher_stats, babip_meta)

    # --- Step 11: Final score ---
    proj_pts = proj_pts_base + pt_result["total_adj"]
    proj_pts = max(0.0, round(proj_pts, 1))

    # --- Step 12: Factor breakdown for dashboard ---
    pitch_diff["adj"]           = pitcher_impact
    pitch_diff["platoon_label"] = platoon_label
    factors = build_factor_breakdown(
        base_pts, platoon_impact, pitch_diff,
        p_adj, form_delta, pt_result["total_adj"],
        babip_meta
    )

    # Stat line display strings
    h_ab_str = f"{stat_line['h']:.1f}H / {stat_line['ab']:.1f}AB"

    # Build scoring rows for the new card layout
    scoring_rows = []
    for cat, proj_val, mult_str, pts_val in [
        ("AB",  stat_line["ab"],      "× −0.50", stat_line["ab"]      * SCORING["AB"]),
        ("1B",  stat_line["singles"], "×  2.80", stat_line["singles"] * SCORING["1B"]),
        ("2B",  stat_line["doubles"], "×  3.80", stat_line["doubles"] * SCORING["2B"]),
        ("3B",  stat_line["triples"], "×  5.70", stat_line["triples"] * SCORING["3B"]),
        ("HR",  stat_line["hr"],      "×  7.50", stat_line["hr"]      * SCORING["HR"]),
        ("R",   stat_line["r"],       "×  0.20", stat_line["r"]       * SCORING["R"]),
        ("RBI", stat_line["rbi"],     "×  0.20", stat_line["rbi"]     * SCORING["RBI"]),
        ("BB",  stat_line["bb"],      "×  1.50", stat_line["bb"]      * SCORING["BB"]),
        ("K",   stat_line["k"],       "× −0.20", stat_line["k"]       * SCORING["K"]),
        ("HBP", stat_line["hbp"],     "×  1.50", stat_line["hbp"]     * SCORING["HBP"]),
    ]:
        scoring_rows.append({
            "cat":  cat,
            "proj": round(proj_val, 2),
            "mult": mult_str,
            "pts":  round(pts_val,  2),
        })

    return {
        "name":         name,
        "mlb_id":       mlb_id,
        "positions":    player["positions"],
        "status":       status,
        "team":         team,
        "opponent":     opponent,
        "venue":        venue,
        "probable_sp":  sp_name,
        "sp_hand":      sp_hand,
        "proj_pts":     proj_pts,
        "confidence":   confidence,
        "stat_line": {
            "h_ab":    h_ab_str,
            "hr":      round(stat_line["hr"],  2),
            "r":       round(stat_line["r"],   2),
            "rbi":     round(stat_line["rbi"], 2),
            "bb":      round(stat_line["bb"],  2),
            "k":       round(stat_line["k"],   2),
        },
        "scoring_rows": scoring_rows,
        "factors":      factors,
        "pitch_rows":   pt_rows,
        "babip_note":   babip_meta.get("direction", ""),
        "data_source":  batter_stats.get("source", "unknown"),
    }


def _infer_team(player):
    """
    Team abbreviation isn't in roster.json — we infer it from
    the matchup data or return empty string as fallback.
    In a future version this could be stored in roster.json directly.
    """
    return ""


def _platoon_adjustment(batter_stats, sp_hand, batter_season,
                        batter_career, name, mlb_id, platoon_splits=None):
    """
    Computes the platoon adjustment using each player's actual L/R splits
    from the MLB Stats API (2022-present aggregated).

    For switch hitters: uses the favorable side vs pitcher hand.
    Falls back to league average if splits unavailable.

    Returns (split_stats_or_None, label_str).
    split_stats: the relevant split row (vL or vR) to use as rate override,
                 or None if falling back to league average.
    """
    bats = batter_stats.get("bats", "R")
    if bats == "S":
        batter_hand = "L" if sp_hand == "R" else "R"
    else:
        batter_hand = bats

    hand_label = "L" if batter_hand == "L" else "R"
    label      = f"Platoon ({hand_label} vs {sp_hand}HP)"

    # Try to use real splits
    if platoon_splits and mlb_id in platoon_splits:
        player_splits = platoon_splits[mlb_id]
        # Which split to use: vs L pitcher or vs R pitcher
        split_key = "vL" if sp_hand == "L" else "vR"
        split_row = player_splits.get(split_key)
        if split_row and split_row.get("pa", 0) >= 20:
            return split_row, label

    # Fallback — league average wOBA delta
    PLATOON_WOBA_DELTA = {
        ("R", "L"):  +0.018,
        ("R", "R"):   0.000,
        ("L", "R"):  +0.025,
        ("L", "L"):  -0.015,
    }
    delta_woba = PLATOON_WOBA_DELTA.get((batter_hand, sp_hand), 0.0)
    # Return None for split_row — caller uses delta_woba via legacy path
    return None, label, delta_woba


def project_all(active_players, il_players, matchups,
                batter_season, batter_career,
                pitcher_season, pitcher_career,
                pitch_splits, pitcher_mixes, platoon_splits=None):
    """
    Runs project_player() for every active/DTD player,
    appends IL players as skeletons, sorts by projected points.

    Returns a list of projection dicts sorted by proj_pts descending.
    DTD and no-game players go to the bottom. IL players last.
    """
    projections = []

    for player in active_players:
        # Find today's matchup for this player's team
        team    = _find_team_for_player(player, matchups)
        matchup = matchups.get(team) if team else None

        if matchup is None and player["status"] != "dtd":
            # No game today — project as 0 pts with a note
            proj = _no_game_skeleton(player)
        else:
            print(f"  Projecting {player['name']}...")
            proj = project_player(
                player, matchup,
                batter_season, batter_career,
                pitcher_season, pitcher_career,
                pitch_splits, pitcher_mixes,
                platoon_splits=platoon_splits
            )
        projections.append(proj)

    # Append IL skeletons at the end
    for player in il_players:
        projections.append({
            "name":       player["name"],
            "mlb_id":     player["mlb_id"],
            "positions":  player["positions"],
            "status":     player["status"],
            "proj_pts":   None,
            "confidence": None,
            "stat_line":  None,
            "factors":    [],
            "pitch_rows": [],
        })

    # Sort: active with projections first (desc pts), then DTD/no-game, then IL
    def sort_key(p):
        if p["status"].startswith("il"):
            return (-1, 0)
        if p["proj_pts"] is None:
            return (0, 0)
        return (1, p["proj_pts"])

    projections.sort(key=sort_key, reverse=True)

    # Add rank numbers (only to players with projections)
    rank = 1
    for p in projections:
        if p["proj_pts"] is not None:
            p["rank"] = rank
            rank += 1
        else:
            p["rank"] = None

    return projections


def _find_team_for_player(player, matchups):
    """
    Returns the team abbreviation for a player using the 'team' field
    in roster.json, then verifies it exists in today's matchups.
    Returns None if the team has no game today.
    """
    team = player.get("team")
    if not team:
        return None
    return team if team in matchups else None


def _no_game_skeleton(player):
    """Returns a projection dict for a player with no game today."""
    return {
        "name":       player["name"],
        "mlb_id":     player["mlb_id"],
        "positions":  player["positions"],
        "status":     "no_game",
        "proj_pts":   None,
        "confidence": None,
        "stat_line":  None,
        "factors":    [],
        "pitch_rows": [],
        "note":       "No game scheduled today",
    }



# ─────────────────────────────────────────────
# SECTION 8 — OUTPUT WRITER + MAIN ENTRY POINT
# ─────────────────────────────────────────────
#
# Writes the final projections to docs/data.json which
# index.html reads to render the dashboard.
#
# data.json schema:
# {
#   "generated_at":  "2025-05-12T09:02:11",
#   "season":        2025,
#   "player_count":  13,
#   "projections":   [ ...projection dicts from Section 7... ]
# }
#
# Functions:
#   write_output(projections)
#   main()
# ─────────────────────────────────────────────


def write_output(projections):
    """
    Serializes projections to docs/data.json.
    Creates the docs/ directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    output = {
        "generated_at":  datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "season":        CURRENT_SEASON,
        "player_count":  len([p for p in projections if p.get("proj_pts") is not None]),
        "projections":   projections,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Written to {OUTPUT_PATH}")
    print(f"  {output['player_count']} players projected")
    print(f"  Generated at {output['generated_at']}")


def main():
    """
    Master entry point. Runs the full pipeline:
      1. Load roster
      2. Get today's matchups
      3. Fetch all data (FanGraphs + Statcast)
      4. Project every player
      5. Write output
    """
    print("=" * 50)
    print("Fantasy Baseball Projections")
    print(f"Date: {datetime.date.today()}")
    print("=" * 50)

    # Step 1 — Roster
    print("\n[1/5] Loading roster...")
    active, il = load_roster()
    print(f"  {len(active)} active/DTD players, {len(il)} on IL")

    # Step 2 — Matchups
    print("\n[2/5] Fetching today's matchups...")
    matchups = get_today_matchups()
    if matchups:
        print(f"  {len(matchups) // 2} games today")
    else:
        print("  No games found or API unavailable")

    # Step 3 — All data
    print("\n[3/5] Fetching stats (this takes ~30-40 seconds)...")
    (batter_season, batter_career,
     pitcher_season, pitcher_career,
     pitch_splits, pitcher_mixes,
     platoon_splits) = fetch_all_data(active, matchups)

    # Step 4 — Projections
    print("\n[4/5] Running projections...")
    projections = project_all(
        active, il, matchups,
        batter_season, batter_career,
        pitcher_season, pitcher_career,
        pitch_splits, pitcher_mixes,
        platoon_splits=platoon_splits
    )

    # Print summary to console
    print("\n  --- Projection Summary ---")
    for p in projections:
        if p["proj_pts"] is not None:
            conf  = p["confidence"]["score"] if p["confidence"] else "?"
            sp    = p.get("probable_sp", "TBD")
            print(f"  #{p['rank']:>2}  {p['name']:<22} {p['proj_pts']:>5.1f} pts  "
                  f"conf={conf}%  vs {sp}")
        else:
            print(f"       {p['name']:<22} —  ({p['status']})")

    # Step 5 — Write output
    print("\n[5/5] Writing output...")
    write_output(projections)

    print("\nDone.")
    print("=" * 50)



# ─────────────────────────────────────────────
# SECTION 9 — ACTUALS FETCHER + LOG WRITER
# ─────────────────────────────────────────────
#
# Runs at the end of every morning's projection job.
# Looks up YESTERDAY'S box scores from the MLB Stats API,
# computes actual fantasy points for each rostered player,
# and appends to docs/projections_log.json.
#
# The log is the source of truth for the accuracy tracker tab
# and the CSV export. It grows by one day each morning and
# never overwrites existing entries.
#
# Log entry schema (one entry per player per day):
# {
#   "date":              "2025-05-11",
#   "player":            "Aaron Judge",
#   "team":              "NYY",
#   "position":          "RF",
#   "opponent":          "BAL",
#   "venue":             "Camden Yards",
#   "probable_sp":       "Corbin Burnes",
#   "sp_hand":           "R",
#   "proj_pts":          21.4,
#   "actual_pts":        18.2,       # null if DNP or no game
#   "diff":              -3.2,       # actual - proj, null if no actual
#   "confidence":        78,
#   "confidence_label":  "high",
#   "did_not_play":      false,
#   "factors": [
#     {"factor": "base_rate",          "adjustment": 14.2},
#     {"factor": "platoon",            "adjustment":  1.8},
#     {"factor": "pitcher_difficulty", "adjustment": -2.1},
#     {"factor": "park_factor",        "adjustment":  0.8},
#     {"factor": "recent_form",        "adjustment":  0.6},
#     {"factor": "pitch_type_breaking","adjustment": -1.1},
#     {"factor": "pitch_type_fastball","adjustment":  0.7},
#   ]
# }
#
# Functions:
#   get_yesterday_boxscores()
#   compute_actual_pts(player_stats_dict)
#   extract_player_game_stats(boxscore, mlb_id)
#   load_log()
#   save_log(log)
#   append_actuals(projections)
#   build_log_entry(projection, actual_stats)
# ─────────────────────────────────────────────

LOG_PATH = os.path.join("docs", "projections_log.json")


def load_log():
    """
    Loads the existing projections log from disk.
    Returns an empty list if the file doesn't exist yet.
    """
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARNING] Could not load log: {e}")
        return []


def save_log(log):
    """
    Saves the projections log to disk.
    Sorts by date descending so newest entries are first.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log.sort(key=lambda x: (x.get("date",""), x.get("player","")), reverse=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved — {len(log)} total entries")


def get_yesterday_boxscores():
    """
    Fetches all game boxscores from yesterday via the MLB Stats API.

    Returns a dict keyed by game_pk:
    {
      game_pk: {
        "home_team": "NYY",
        "away_team": "BAL",
        "venue":     "Yankee Stadium",
        "players": {
          mlb_id: {
            "ab": 4, "h": 2, "singles": 1, "doubles": 1,
            "triples": 0, "hr": 0, "r": 1, "rbi": 1,
            "bb": 0, "k": 1, "hbp": 0, "sb": 0, "cs": 0,
            "gidp": 0, "sac": 0, "played": True
          },
          ...
        }
      }
    }
    Returns empty dict if API unavailable or no games yesterday.
    """
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{MLB_API}/schedule?sportId=1&date={yesterday}&hydrate=boxscore,venue,team"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARNING] MLB API boxscore fetch failed: {e}")
        return {}

    games = {}

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk   = game.get("gamePk")
            venue     = game.get("venue", {}).get("name", "Unknown")
            home_team = game["teams"]["home"]["team"].get("abbreviation","")
            away_team = game["teams"]["away"]["team"].get("abbreviation","")

            # Fetch full boxscore for this game
            box_url = f"{MLB_API}/game/{game_pk}/boxscore"
            try:
                box_resp = requests.get(box_url, timeout=10)
                box_resp.raise_for_status()
                box_data = box_resp.json()
            except Exception as e:
                print(f"  [WARNING] Boxscore fetch failed for game {game_pk}: {e}")
                continue

            players = {}

            for side in ("home", "away"):
                team_players = box_data.get("teams", {}).get(side, {}).get("players", {})
                for player_key, player_data in team_players.items():
                    pid    = player_data.get("person", {}).get("id")
                    stats  = player_data.get("stats", {}).get("batting", {})
                    if not pid or not stats:
                        continue

                    # Only include batters who actually appeared
                    ab = int(stats.get("atBats", 0))
                    h  = int(stats.get("hits",   0))

                    # Derive hit types
                    hr      = int(stats.get("homeRuns",     0))
                    doubles = int(stats.get("doubles",      0))
                    triples = int(stats.get("triples",      0))
                    singles = max(h - hr - doubles - triples, 0)

                    players[pid] = {
                        "ab":      ab,
                        "h":       h,
                        "singles": singles,
                        "doubles": doubles,
                        "triples": triples,
                        "hr":      hr,
                        "r":       int(stats.get("runs",              0)),
                        "rbi":     int(stats.get("rbi",               0)),
                        "bb":      int(stats.get("baseOnBalls",       0)),
                        "k":       int(stats.get("strikeOuts",        0)),
                        "hbp":     int(stats.get("hitByPitch",        0)),
                        "sb":      int(stats.get("stolenBases",       0)),
                        "cs":      int(stats.get("caughtStealing",    0)),
                        "gidp":    int(stats.get("groundIntoDoublePlay", 0)),
                        "sac":     int(stats.get("sacBunts",          0))
                                 + int(stats.get("sacFlies",          0)),
                        "played":  True,
                    }

            games[game_pk] = {
                "home_team": home_team,
                "away_team": away_team,
                "venue":     venue,
                "players":   players,
            }

    print(f"  Fetched boxscores for {len(games)} games on {yesterday}")
    return games


def compute_actual_pts(s):
    """
    Converts a player's actual stat line into fantasy points
    using the exact league scoring from SCORING config.

    s — dict with keys: ab, singles, doubles, triples, hr,
                        r, rbi, bb, k, hbp, sac, sb, cs, gidp
    Returns float of total fantasy points.
    """
    pts = (
        s.get("ab",      0) * SCORING["AB"]   +
        s.get("singles", 0) * SCORING["1B"]   +
        s.get("doubles", 0) * SCORING["2B"]   +
        s.get("triples", 0) * SCORING["3B"]   +
        s.get("hr",      0) * SCORING["HR"]   +
        s.get("r",       0) * SCORING["R"]    +
        s.get("rbi",     0) * SCORING["RBI"]  +
        s.get("bb",      0) * SCORING["BB"]   +
        s.get("k",       0) * SCORING["K"]    +
        s.get("hbp",     0) * SCORING["HBP"]  +
        s.get("sac",     0) * SCORING["SAC"]  +
        s.get("sb",      0) * SCORING["SB"]   +
        s.get("cs",      0) * SCORING["CS"]   +
        s.get("gidp",    0) * SCORING["GIDP"]
    )
    return round(pts, 2)


def find_player_in_boxscores(mlb_id, boxscores):
    """
    Searches all yesterday's boxscores for a player by MLB ID.
    Returns (game_dict, player_stats) or (None, None) if not found.
    """
    for game_pk, game in boxscores.items():
        if mlb_id in game["players"]:
            return game, game["players"][mlb_id]
    return None, None


def _extract_factors(projection):
    """Extracts tidy factor list from a projection dict."""
    factors = []
    for f in projection.get("factors", []):
        label = f.get("label", "")
        adj   = f.get("adj")
        if adj is None:
            continue
        if "base rate"            in label.lower(): key = "base_rate"
        elif "platoon"            in label.lower(): key = "platoon"
        elif "pitcher"            in label.lower(): key = "pitcher_difficulty"
        elif "park factor"        in label.lower(): key = "park_factor"
        elif "recent form"        in label.lower(): key = "recent_form"
        elif "babip"              in label.lower(): key = "babip_adjustment"
        else:                                        key = label.lower().replace(" ","_")
        factors.append({"factor": key, "adjustment": round(float(adj), 2)})
    for row in projection.get("pitch_rows", []):
        bucket = row.get("label","").lower().replace(" ","_")
        factors.append({
            "factor":     f"pitch_type_{bucket}",
            "adjustment": round(float(row.get("adj", 0)), 2),
        })
    return factors


def build_projection_log_entry(projection, today_str):
    """
    Builds a log entry for TODAY's projection with actual_pts=null.
    Called at projection time — actuals filled in tomorrow.
    """
    conf      = projection.get("confidence") or {}
    positions = projection.get("positions", [])
    position  = positions[0] if positions else ""

    return {
        "date":             today_str,
        "player":           projection.get("name", ""),
        "mlb_id":           projection.get("mlb_id"),
        "team":             projection.get("team", ""),
        "position":         position,
        "opponent":         projection.get("opponent", ""),
        "venue":            projection.get("venue", ""),
        "probable_sp":      projection.get("probable_sp", ""),
        "sp_hand":          projection.get("sp_hand", ""),
        "proj_pts":         projection.get("proj_pts"),
        "actual_pts":       None,
        "diff":             None,
        "confidence":       conf.get("score"),
        "confidence_label": conf.get("label"),
        "did_not_play":     False,
        "factors":          _extract_factors(projection),
    }


def log_today_projections(projections):
    """
    Step A — called right after write_output().
    Appends today's projections to the log with actual_pts=null.
    Skips players with no projection (DTD, IL, no game).
    Skips if today's entries already exist (idempotent).
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    log       = load_log()

    # Check for duplicate
    existing_dates = {entry["date"] for entry in log}
    if today_str in existing_dates:
        print(f"  Projections for {today_str} already in log — skipping")
        return

    new_entries = []
    for proj in projections:
        if proj.get("proj_pts") is None:
            continue
        if proj.get("status") in ("dtd", "no_game") or            (proj.get("status") or "").startswith("il"):
            continue
        new_entries.append(build_projection_log_entry(proj, today_str))

    log.extend(new_entries)
    save_log(log)
    print(f"  Logged {len(new_entries)} projections for {today_str}")


def fill_yesterday_actuals():
    """
    Step B — called after log_today_projections().
    Finds yesterday's log entries where actual_pts is null,
    fetches boxscores, and fills in the actuals.
    """
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n[9b] Filling actuals for {yesterday}...")

    log = load_log()

    # Find yesterday's entries that need actuals filled in
    pending = [
        (i, entry) for i, entry in enumerate(log)
        if entry.get("date") == yesterday and entry.get("actual_pts") is None
        and not entry.get("did_not_play", False)
    ]

    if not pending:
        print(f"  No pending entries for {yesterday}")
        return

    print(f"  Found {len(pending)} entries needing actuals")

    # Fetch boxscores
    boxscores = get_yesterday_boxscores()
    if not boxscores:
        print("  No boxscores found — skipping")
        return

    filled = 0
    for idx, entry in pending:
        mlb_id = entry.get("mlb_id")
        if not mlb_id:
            continue

        game, actual_stats = find_player_in_boxscores(mlb_id, boxscores)

        if actual_stats and actual_stats.get("played"):
            actual_pts = compute_actual_pts(actual_stats)
            proj_pts   = entry.get("proj_pts")
            diff       = round(actual_pts - proj_pts, 2) if proj_pts is not None else None
            log[idx]["actual_pts"]   = actual_pts
            log[idx]["diff"]         = diff
            log[idx]["did_not_play"] = False
            filled += 1
            diff_str = f"{diff:+.1f}" if diff is not None else "—"
            print(f"  {entry['player']:<22} proj={proj_pts:.1f}  actual={actual_pts:.1f}  diff={diff_str}")
        else:
            # Player had no game or didn't play
            log[idx]["did_not_play"] = True
            print(f"  {entry['player']:<22} DNP or no game")

    save_log(log)
    print(f"  Filled actuals for {filled}/{len(pending)} players")


def append_actuals(projections):
    """
    Master Section 9 function — now a thin wrapper that calls
    the two-step process: log today's projections, fill yesterday's actuals.
    """
    print("\n[9] Updating projections log...")
    log_today_projections(projections)
    fill_yesterday_actuals()


# ─────────────────────────────────────────────
# END SECTION 9
# ─────────────────────────────────────────────







