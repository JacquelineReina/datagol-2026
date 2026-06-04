from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

LABELS = np.array(["H", "D", "A"])
FEATURES = [
    "elo_diff",
    "gf10_diff",
    "ga10_diff",
    "points10_diff",
    "goal_diff10_diff",
    "rest_diff",
    "home_adv",
    "tournament_weight",
    "experience_diff",
]
MAX_GOALS = 8


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def temperature_scale(probs: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temp)
    return softmax(logits)


class PortableMultinomialLogit:
    """Inference-only multinomial logistic regression without sklearn/joblib."""

    def __init__(self, artifact: dict):
        self.classes = [str(x) for x in artifact["classes"]]
        self.mean = np.asarray(artifact["scaler_mean"], dtype=float)
        self.scale = np.asarray(artifact["scaler_scale"], dtype=float)
        self.coef = np.asarray(artifact["coef"], dtype=float)
        self.intercept = np.asarray(artifact["intercept"], dtype=float)

    def predict_proba(self, row: pd.DataFrame) -> np.ndarray:
        x = row[FEATURES].iloc[0].to_numpy(dtype=float)
        scale = np.where(self.scale == 0.0, 1.0, self.scale)
        z = (x - self.mean) / scale
        logits = self.coef @ z + self.intercept
        return softmax(logits)


class FrozenPredictor:
    def __init__(self):
        self.meta = json.loads(
            (MODELS / "production_metadata.json").read_text(encoding="utf-8")
        )
        self.dc = json.loads(
            (MODELS / "dixon_coles.json").read_text(encoding="utf-8")
        )
        self.state = json.loads(
            (MODELS / "inference_state.json").read_text(encoding="utf-8")
        )
        logit_artifact = json.loads(
            (MODELS / "elo_logit_portable.json").read_text(encoding="utf-8")
        )
        self.logit = PortableMultinomialLogit(logit_artifact)

        self.teams = self.dc["teams"]
        self.idx = {team: index for index, team in enumerate(self.teams)}
        self.attack = np.asarray(self.dc["attack"], dtype=float)
        self.defence = np.asarray(self.dc["defence"], dtype=float)

    def _team_state(self, team: str) -> dict:
        avg = float(self.state["global_avg_goals_per_team"])
        return self.state["teams"].get(
            team,
            {
                "elo": 1500.0,
                "n_matches": 0,
                "last_date": None,
                "gf10": avg,
                "ga10": avg,
                "points10": 1.0,
                "goal_diff10": 0.0,
            },
        )

    def _dc_lambdas(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
    ):
        index_a = self.idx.get(team_a)
        index_b = self.idx.get(team_b)

        attack_a = self.attack[index_a] if index_a is not None else 0.0
        defence_a = self.defence[index_a] if index_a is not None else 0.0
        attack_b = self.attack[index_b] if index_b is not None else 0.0
        defence_b = self.defence[index_b] if index_b is not None else 0.0

        intercept = float(self.dc["intercept"])
        home = float(self.dc["home_adv_log"])

        lambda_a = math.exp(intercept + attack_a - defence_b + (home if host_a else 0.0))
        lambda_b = math.exp(intercept + attack_b - defence_a + (home if host_b else 0.0))

        return (
            float(np.clip(lambda_a, 0.03, 5.5)),
            float(np.clip(lambda_b, 0.03, 5.5)),
        )

    def score_matrix(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
    ):
        lambda_a, lambda_b = self._dc_lambdas(team_a, team_b, host_a, host_b)
        rho = float(self.dc["rho"])
        goals = np.arange(MAX_GOALS + 1)

        matrix = np.outer(
            np.exp(goals * np.log(lambda_a) - lambda_a - gammaln(goals + 1)),
            np.exp(goals * np.log(lambda_b) - lambda_b - gammaln(goals + 1)),
        )

        matrix[0, 0] *= 1 - lambda_a * lambda_b * rho
        matrix[0, 1] *= 1 + lambda_a * rho
        matrix[1, 0] *= 1 + lambda_b * rho
        matrix[1, 1] *= 1 - rho

        matrix = np.clip(matrix, 1e-14, None)
        matrix /= matrix.sum()
        return matrix, lambda_a, lambda_b

    def _features(
        self,
        team_a: str,
        team_b: str,
        host_a: bool,
        host_b: bool,
        rest_a: int,
        rest_b: int,
    ) -> pd.DataFrame:
        state_a = self._team_state(team_a)
        state_b = self._team_state(team_b)

        row = {
            "elo_diff": (float(state_a["elo"]) - float(state_b["elo"])) / 400.0,
            "gf10_diff": float(state_a["gf10"]) - float(state_b["gf10"]),
            "ga10_diff": float(state_a["ga10"]) - float(state_b["ga10"]),
            "points10_diff": float(state_a["points10"]) - float(state_b["points10"]),
            "goal_diff10_diff": float(state_a["goal_diff10"]) - float(state_b["goal_diff10"]),
            "rest_diff": float(np.clip(rest_a - rest_b, -10, 10)) / 10.0,
            "home_adv": float(host_a) - float(host_b),
            "tournament_weight": 1.0,
            "experience_diff": float(
                np.tanh((int(state_a["n_matches"]) - int(state_b["n_matches"])) / 50.0)
            ),
        }
        return pd.DataFrame([row], columns=FEATURES)

    def predict(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
        rest_a: int = 7,
        rest_b: int = 7,
    ) -> dict:
        matrix, lambda_a, lambda_b = self.score_matrix(
            team_a,
            team_b,
            host_a,
            host_b,
        )
        dixon_coles = np.array(
            [
                np.tril(matrix, -1).sum(),
                np.trace(matrix),
                np.triu(matrix, 1).sum(),
            ],
            dtype=float,
        )

        row = self._features(team_a, team_b, host_a, host_b, rest_a, rest_b)
        raw_logit = self.logit.predict_proba(row)
        class_to_prob = dict(zip(self.logit.classes, raw_logit))
        ordered_logit = np.array(
            [class_to_prob[label] for label in LABELS],
            dtype=float,
        )
        elo_logit = temperature_scale(
            ordered_logit,
            float(self.meta["logit_temperature"]),
        )

        # Gradient Boosting was trained and validated, but its optimized production
        # weight is 0 %. It is intentionally not loaded during inference.
        weights = self.meta["ensemble_weights"]
        ensemble = (
            float(weights["dixon_coles"]) * dixon_coles
            + float(weights["elo_logit"]) * elo_logit
        )
        ensemble = ensemble / ensemble.sum()

        scores = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                scores.append({"score": f"{i}-{j}", "prob": float(matrix[i, j])})

        scores.sort(key=lambda item: item["prob"], reverse=True)

        return {
            "teams": [team_a, team_b],
            "host_flags": [host_a, host_b],
            "rest_days": [rest_a, rest_b],
            "expected_goals": [lambda_a, lambda_b],
            "probabilities": {
                "dixon_coles": dixon_coles.tolist(),
                "elo_logit": elo_logit.tolist(),
                "gradient_boosting": None,
                "ensemble": ensemble.tolist(),
            },
            "top5_scores": scores[:5],
        }


def load_fixtures():
    return pd.read_csv(DATA / "worldcup_2026_group_stage.csv", parse_dates=["date"])


def venue_country(ground: str) -> str:
    mexico = {"Mexico City", "Guadalajara (Zapopan)", "Monterrey (Guadalupe)"}
    canada = {"Toronto", "Vancouver"}
    return "Mexico" if ground in mexico else ("Canada" if ground in canada else "United States")


def fixture_context(fixtures: pd.DataFrame, index: int):
    row = fixtures.iloc[index]
    team_a = str(row.team_a)
    team_b = str(row.team_b)
    ground = str(row.ground)
    date = row.date
    country = venue_country(ground)

    host_a = team_a == country
    host_b = team_b == country

    previous = {}
    before = fixtures[fixtures.date < date]
    for team in [team_a, team_b]:
        played = before[(before.team_a == team) | (before.team_b == team)]
        previous[team] = 7 if played.empty else int((date - played.date.max()).days)

    return {
        "row": row,
        "team_a": team_a,
        "team_b": team_b,
        "host_a": host_a,
        "host_b": host_b,
        "rest_a": previous[team_a],
        "rest_b": previous[team_b],
        "venue_country": country,
    }
