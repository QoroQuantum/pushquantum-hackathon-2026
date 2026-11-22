# QIcing portfolio attack | method: partitioned-qaoa + repair + local search
# settings: routine=qaoa cluster=12 n_layers=2 penalty_weight=100.0 mc_pop=50 mc_iter=10 seed=1 circuits=31062
# quantum: True | qubo vars: 630 | max qubits: 12
instance po_a010_t10_orig
budget 4
lambda 0.0001
objective -54923
# period symbol long short
0 AVGO 1 0
0 GOOGL 3 0
1 META 1 0
1 TSLA 0 2
1 WMT 0 1
2 META 3 0
2 NVDA 0 1
3 AMZN 0 1
3 AVGO 0 1
3 GOOG 0 1
3 META 1 0
4 AAPL 1 0
4 GOOG 1 0
4 META 1 0
4 TSLA 0 1
5 GOOGL 1 0
5 META 0 1
5 TSLA 2 0
6 GOOGL 1 0
6 NVDA 0 1
6 TSLA 1 0
6 WMT 1 0
7 AAPL 0 1
7 AVGO 0 1
7 NVDA 1 0
7 WMT 1 0
8 AAPL 1 0
8 GOOG 1 0
8 WMT 0 2
