"""
Step 2: Feature engineering — reads raw JSON data and builds train/test CSVs.

Reads from data/raw/:
  - games_february.json
  - games_march.json
  - player_stats.json

Features (all known before the game starts):
  - rating_diff: white_rating - black_rating
  - round: tournament round number (1-11)
  - win_rate_diff: white blitz win rate - black blitz win rate
  - draw_rate_diff: white blitz draw rate - black blitz draw rate
  - total_games_diff: white total blitz games - black total blitz games
  - best_rating_diff: white best blitz rating - black best blitz rating

Target:
  - outcome: 'win' / 'draw' / 'loss' (from White's perspective)

Output (saved to data/dataset/):
  - train.csv (February tournament)
  - test.csv (March tournament)
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DATASET_DIR = Path(__file__).parent.parent / "data" / "dataset"

# Chess.com result strings → outcome from White's perspective
WHITE_WIN_RESULTS = {"win"}
DRAW_RESULTS = {
    "agreed", "stalemate", "repetition", "50move",
    "insufficient", "timevsinsufficient",
}


def load_json(filename: str):
    with open(RAW_DIR / filename) as f:
        return json.load(f)


def get_outcome_for_white(white_result: str) -> str:
    """Map Chess.com result string to outcome from White's perspective."""
    if white_result in WHITE_WIN_RESULTS:
        return "win"
    if white_result in DRAW_RESULTS:
        return "draw"
    return "loss"


def extract_player_blitz_stats(player_stats: dict, username: str) -> dict:
    """Extract blitz win rate, draw rate, total games, and best rating."""
    stats = player_stats.get(username, {})
    blitz = stats.get("chess_blitz", {})
    record = blitz.get("record", {})

    wins = record.get("win", 0)
    losses = record.get("loss", 0)
    draws = record.get("draw", 0)
    total = wins + losses + draws

    best = blitz.get("best", {}).get("rating", 0)

    return {
        "win_rate": wins / total if total > 0 else 0,
        "draw_rate": draws / total if total > 0 else 0,
        "total_games": total,
        "best_rating": best,
    }


def build_game_features(game: dict, player_stats: dict) -> dict | None:
    """Extract all features from a single game."""
    white = game.get("white", {})
    black = game.get("black", {})

    white_rating = white.get("rating")
    black_rating = black.get("rating")
    white_result = white.get("result")

    if white_rating is None or black_rating is None or white_result is None:
        return None

    white_username = white.get("username", "").lower()
    black_username = black.get("username", "").lower()

    white_stats = extract_player_blitz_stats(player_stats, white_username)
    black_stats = extract_player_blitz_stats(player_stats, black_username)

    return {
        # Identifiers
        "white_username": white_username,
        "black_username": black_username,
        # Features
        "rating_diff": white_rating - black_rating,
        "round": game.get("_round"),
        "win_rate_diff": white_stats["win_rate"] - black_stats["win_rate"],
        "draw_rate_diff": white_stats["draw_rate"] - black_stats["draw_rate"],
        "total_games_diff": white_stats["total_games"] - black_stats["total_games"],
        "best_rating_diff": white_stats["best_rating"] - black_stats["best_rating"],
        # Target
        "outcome": get_outcome_for_white(white_result),
    }


def build_dataset(games: list[dict], player_stats: dict, label: str) -> pd.DataFrame:
    """Build a DataFrame of features from a list of games."""
    rows = []
    for game in tqdm(games, desc=f"  Features ({label})"):
        features = build_game_features(game, player_stats)
        if features:
            rows.append(features)
    return pd.DataFrame(rows)


def main():
    # Load raw data
    print("Loading raw data...")
    games_feb = load_json("games_february.json")
    games_mar = load_json("games_march.json")
    player_stats = load_json("player_stats.json")
    print(f"  February games: {len(games_feb)}")
    print(f"  March games:    {len(games_mar)}")
    print(f"  Player stats:   {len(player_stats)}")

    # Build features
    print("\nBuilding features...")
    train_df = build_dataset(games_feb, player_stats, "february")
    test_df = build_dataset(games_mar, player_stats, "march")

    # Summary
    print(f"\nTrain set: {len(train_df)} games")
    print(f"Test set:  {len(test_df)} games")
    print(f"\nOutcome distribution (train):\n{train_df['outcome'].value_counts().to_string()}")
    print(f"\nOutcome distribution (test):\n{test_df['outcome'].value_counts().to_string()}")

    # Save
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATASET_DIR / "train.csv", index=False)
    test_df.to_csv(DATASET_DIR / "test.csv", index=False)
    print(f"\nSaved: data/dataset/train.csv ({len(train_df)} rows)")
    print(f"Saved: data/dataset/test.csv ({len(test_df)} rows)")

    # Feature overview
    print("\n--- Feature Summary (train) ---")
    feature_cols = [c for c in train_df.columns if c not in ("white_username", "black_username", "outcome")]
    print(train_df[feature_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
