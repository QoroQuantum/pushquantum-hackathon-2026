# QIcing portfolio attack | method: partitioned-qaoa + repair
# settings: routine=qaoa cluster=12 n_layers=2 penalty_weight=100.0 mc_pop=50 mc_iter=10 seed=1 circuits=31062
# quantum: True | qubo vars: 630 | max qubits: 12
instance po_a010_t10_orig
budget 4
lambda 0.0001
objective -3484
# period symbol long short
0 AVGO 1 0
0 GOOGL 1 0
0 MSFT 0 1
0 WMT 0 1
1 AMZN 1 0
1 AVGO 1 0
1 TSLA 0 1
1 WMT 1 0
2 MSFT 1 0
2 NVDA 0 1
3 AMZN 0 1
3 NVDA 1 0
4 AAPL 1 0
4 AVGO 0 1
4 GOOG 1 0
4 META 1 0
5 AAPL 2 0
5 GOOGL 1 0
5 TSLA 1 0
6 AAPL 1 0
6 GOOGL 1 0
6 TSLA 1 0
6 WMT 1 0
7 AAPL 0 1
7 AVGO 0 1
7 GOOG 0 1
7 MSFT 1 0
8 GOOG 1 0
8 GOOGL 1 0
8 WMT 0 2
9 AMZN 0 1
9 GOOGL 0 1
9 NVDA 0 1
9 TSLA 0 1
