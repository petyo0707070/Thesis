import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv('X_1.csv')
y = pd.read_csv('y_1.csv')


#y['Adjustment'] = y['Realistic PNL'] - y['Bid Ask PNL']
#y['Realistic PNL'] = y['Realistic PNL'] + y['Adjustment']

X['Realistic PNL'] = y['Realistic PNL'].values
X['Market Cap Percentile'] = pd.qcut(X['Log Market Cap'], 100, labels = False) + 1
avg_pnl_by_marketcap = X.groupby('Market Cap Percentile')['Realistic PNL'].mean()


X['Backtest PNL Percentile'] = pd.qcut(X['PNL Realistic (8)'], 100, labels = False) + 1
avg_pnl_by_backtest_pnl = X.groupby('Backtest PNL Percentile')['Realistic PNL'].mean()

X['Implied Move Percentile'] = pd.qcut(X['Implied Move'], 100, labels = False) + 1
avg_pnl_by_implied_move = X.groupby('Implied Move Percentile')['Realistic PNL'].mean()

X['Realized Move Pct Abs (1) Percentile'] = pd.qcut(X['Realized Move Pct Abs (1)'], 100, labels = False) + 1
avg_pnl_by_prvious_move = X.groupby('Realized Move Pct Abs (1) Percentile')['Realistic PNL'].mean()


X['Composite Percentile'] = (X['Market Cap Percentile'] + X['Backtest PNL Percentile'] + X['Implied Move Percentile'] + X['Realized Move Pct Abs (1) Percentile']) / 4
avg_pnl_by_composite =  X.groupby('Composite Percentile')['Realistic PNL'].mean()



plt.figure(figsize=(10, 5))
plt.plot(avg_pnl_by_marketcap.index, avg_pnl_by_marketcap.values)
plt.plot(avg_pnl_by_marketcap.index, np.polyval(np.polyfit(avg_pnl_by_marketcap.index, avg_pnl_by_marketcap.values, 1), avg_pnl_by_marketcap.index,), label="Best Fit Line", linestyle="--")
plt.xlabel("Markcap Percentile")
plt.ylabel("Straddle PNL")
plt.title("Marketcap vs Straddle PNL")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(avg_pnl_by_backtest_pnl.index, avg_pnl_by_backtest_pnl.values)
plt.plot(avg_pnl_by_backtest_pnl.index, np.polyval(np.polyfit(avg_pnl_by_backtest_pnl.index, avg_pnl_by_backtest_pnl.values, 1), avg_pnl_by_backtest_pnl.index,), label="Best Fit Line", linestyle="--")
plt.xlabel("PNL Backtest Percentile")
plt.ylabel("Straddle PNL")
plt.title("Straddle Backtest PNL vs Straddle PNL")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(avg_pnl_by_implied_move.index, avg_pnl_by_implied_move.values)
plt.plot(avg_pnl_by_implied_move.index, np.polyval(np.polyfit(avg_pnl_by_implied_move.index, avg_pnl_by_implied_move.values, 1), avg_pnl_by_implied_move.index,), label="Best Fit Line", linestyle="--")
plt.xlabel("Implied Move Percentile")
plt.ylabel("Straddle PNL")
plt.title("Implied Move vs Straddle PNL")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(avg_pnl_by_prvious_move.index, avg_pnl_by_prvious_move.values)
plt.plot(avg_pnl_by_prvious_move.index, np.polyval(np.polyfit(avg_pnl_by_prvious_move.index, avg_pnl_by_prvious_move.values, 1), avg_pnl_by_prvious_move.index,), label="Best Fit Line", linestyle="--")
plt.xlabel("Previous Earnings Move Abs Percentile")
plt.ylabel("Straddle PNL")
plt.title("Previous Earnings Move vs Straddle PNL")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(avg_pnl_by_composite.index, avg_pnl_by_composite.values)
plt.plot(avg_pnl_by_composite.index, np.polyval(np.polyfit(avg_pnl_by_composite.index, avg_pnl_by_composite.values, 1), avg_pnl_by_composite.index,), label="Best Fit Line", linestyle="--")
plt.xlabel("Composite Percentile")
plt.ylabel("Straddle PNL")
plt.title("Composite Percentile vs Straddle PNL")
plt.show()


print(X)

