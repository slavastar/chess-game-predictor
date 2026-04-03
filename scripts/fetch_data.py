"""
Step 1: Fetch all raw data from Chess.com API and save to JSON files.

Fetches:
  - Tournament metadata (2 requests)
  - Round + group game data (11 rounds × 2 tournaments)
  - Player stats for all unique players (~800 requests)

Output (saved to data/):
  - tournament_february.json
  - tournament_march.json
  - games_february.json
  - games_march.json
  - player_stats.json
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm

HEADERS = {
    "User-Agent": "ChessOutcomePrediction/1.0 (github: @slavastar)"
}

BASE_URL = "https://api.chess.com/pub"

TOURNAMENTS = {
    "february": "titled-tuesday-blitz-february-10-2026-6221327",
    "march": "titled-tuesday-blitz-march-10-2026-6277141",
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON from Chess.com API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  Failed: {url} — {e}")
                return None
    return None


def save_json(data, filename: str):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def fetch_tournament_info(tournament_id: str) -> dict | None:
    """Fetch tournament metadata."""
    return fetch_json(f"{BASE_URL}/tournament/{tournament_id}")


def fetch_all_games(tournament: dict, label: str) -> list[dict]:
    """Traverse rounds and groups to collect all games for a tournament."""
    all_games = []
    rounds = tournament["rounds"]

    for round_idx, round_url in tqdm(
        enumerate(rounds, start=1), total=len(rounds), desc=f"  Rounds ({label})"
    ):
        round_data = fetch_json(round_url)
        if not round_data:
            continue

        for group_url in round_data.get("groups", []):
            group_data = fetch_json(group_url)
            if not group_data:
                continue

            for game in group_data.get("games", []):
                game["_round"] = round_idx
                all_games.append(game)

        time.sleep(0.2)

    return all_games


def collect_usernames(games: list[dict]) -> set[str]:
    """Extract all unique usernames from a list of games."""
    usernames = set()
    for game in games:
        white = game.get("white", {}).get("username", "")
        black = game.get("black", {}).get("username", "")
        if white:
            usernames.add(white.lower())
        if black:
            usernames.add(black.lower())
    return usernames


def fetch_all_player_stats(usernames: set[str]) -> dict:
    """Fetch blitz stats for all players."""
    stats = {}
    for username in tqdm(sorted(usernames), desc="  Player stats"):
        data = fetch_json(f"{BASE_URL}/player/{username}/stats")
        if data:
            stats[username] = data
        time.sleep(0.1)
    return stats


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Fetch tournament metadata
    print("Fetching tournament metadata...")
    all_games = []

    for label, tid in TOURNAMENTS.items():
        print(f"\n--- {label.upper()} ---")
        tournament = fetch_tournament_info(tid)
        if not tournament:
            print(f"  Failed to fetch tournament: {tid}")
            continue
        save_json(tournament, f"tournament_{label}.json")

        # 2. Fetch all games
        games = fetch_all_games(tournament, label)
        save_json(games, f"games_{label}.json")
        print(f"  Games: {len(games)}")
        all_games.extend(games)

    # 3. Fetch player stats for all unique players
    print("\n--- PLAYER STATS ---")
    usernames = collect_usernames(all_games)
    print(f"  Unique players: {len(usernames)}")
    player_stats = fetch_all_player_stats(usernames)
    save_json(player_stats, "player_stats.json")
    print(f"  Stats fetched: {len(player_stats)}")

    print("\nDone! All data saved to data/")


if __name__ == "__main__":
    main()
