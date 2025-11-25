"""
Package Import
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw["Adj Close"]

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=15, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        # Add a secondary longer lookback for trend detection
        self.long_lookback = lookback * 3

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=self.price.index, columns=self.price.columns)

        """
        TODO: Complete Task 4 Below
        """
        # Multi-timeframe strategy with momentum and trend filtering
        for i in range(max(self.lookback, self.long_lookback) + 1, len(self.price)):
            R_n = self.returns.copy()[assets].iloc[i - self.lookback : i]
            R_long = self.returns.copy()[assets].iloc[i - self.long_lookback : i]

            # Short-term momentum
            short_momentum = R_n.mean()
            short_vol = R_n.std()
            short_sharpe = short_momentum / (short_vol + 1e-8)

            # Long-term trend
            long_momentum = R_long.mean()

            # Select assets with both positive short-term Sharpe and long-term momentum
            selected_assets = []
            for asset in assets:
                if short_sharpe[asset] > 0 and long_momentum[asset] > 0:
                    selected_assets.append(asset)

            if len(selected_assets) >= 3:
                # Use mean-variance optimization on selected assets
                R_filtered = R_n[selected_assets]
                weights_filtered = self.mv_opt(R_filtered, self.gamma)

                # Assign weights
                for j, asset in enumerate(selected_assets):
                    self.portfolio_weights.loc[self.price.index[i], asset] = weights_filtered[j]

                # Set other assets to 0
                for asset in assets:
                    if asset not in selected_assets:
                        self.portfolio_weights.loc[self.price.index[i], asset] = 0
            else:
                # If less than 3 assets pass the filter, use positive short-term Sharpe only
                positive_sharpe_assets = short_sharpe[short_sharpe > 0].index.tolist()
                if len(positive_sharpe_assets) >= 3:
                    R_filtered = R_n[positive_sharpe_assets]
                    weights_filtered = self.mv_opt(R_filtered, self.gamma)
                    for j, asset in enumerate(positive_sharpe_assets):
                        self.portfolio_weights.loc[self.price.index[i], asset] = weights_filtered[j]
                    for asset in assets:
                        if asset not in positive_sharpe_assets:
                            self.portfolio_weights.loc[self.price.index[i], asset] = 0
                else:
                    # Use all assets
                    weights = self.mv_opt(R_n, self.gamma)
                    self.portfolio_weights.loc[self.price.index[i], assets] = weights

        # Set excluded asset weight to 0
        self.portfolio_weights[self.exclude] = 0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        """Mean-Variance Optimization using Gurobi"""
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # Decision variables: portfolio weights
                w = model.addMVar(n, name="w", lb=0, ub=1)

                # Objective: maximize mean return - (gamma/2) * variance
                portfolio_return = mu @ w
                portfolio_variance = w @ Sigma @ w

                model.setObjective(portfolio_return - (gamma / 2) * portfolio_variance, gp.GRB.MAXIMIZE)

                # Constraints: sum(w) = 1 (no leverage)
                model.addConstr(w.sum() == 1, name="budget")

                model.optimize()

                # Extract solution
                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution.append(var.X)
                    return solution
                else:
                    # Return equal weights if optimization fails
                    return [1.0 / n] * n

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge

    parser = argparse.ArgumentParser(description="Introduction to Fintech Assignment 3 Part 12")

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument("--report", action="append", help="Report for evaluation metric")

    parser.add_argument("--cumulative", action="append", help="Cumulative product result")

    args = parser.parse_args()

    judge = AssignmentJudge()

    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
