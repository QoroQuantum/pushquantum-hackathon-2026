# QIcing portfolio attack | method: QLNS-pce (quantum sub-solver, Maestro)
# settings: routine=pce n_layers=3 window_assets=5 p_time=0.5 penalty_weight=5000 mc_pop=40 mc_iter=8 shots=4000 iters=400 seed=4 start=empty warm_start=none char_job= circuits=128400
# quantum: True | qubo vars: 630 | max qubits: 9
instance po_a010_t10_orig
budget 4
lambda 0.0001
objective -68785
# period symbol long short
0 AAPL 0 1
0 GOOG 1 0
0 GOOGL 2 0
1 META 1 0
1 TSLA 0 2
1 WMT 0 1
2 AVGO 1 0
2 META 2 0
2 NVDA 0 1
3 AVGO 0 1
3 GOOG 0 2
3 META 1 0
4 AMZN 1 0
4 META 2 0
4 TSLA 0 1
5 GOOG 1 0
5 META 0 1
5 TSLA 2 0
6 GOOG 1 0
6 MSFT 1 0
6 NVDA 0 1
6 TSLA 1 0
7 NVDA 1 0
7 TSLA 0 1
7 WMT 2 0
8 AAPL 3 0
8 AVGO 1 0
