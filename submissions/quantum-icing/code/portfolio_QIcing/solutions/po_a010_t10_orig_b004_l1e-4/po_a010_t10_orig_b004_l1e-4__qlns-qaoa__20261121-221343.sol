# QIcing portfolio attack | method: QLNS-qaoa (quantum sub-solver, Maestro)
# settings: routine=qaoa n_layers=4 window_assets=2 p_time=0.3 penalty_weight=5000 mc_pop=30 mc_iter=5 shots=4000 iters=200 seed=1 start=solutions/po_a010_t10_orig_b004_l1e-4/po_a010_t10_orig_b004_l1e-4__part-qaoa-ls-s1__20261121-215323.sol warm_start=char-service char_job=c3a51b2e-ad3f-4cae-924b-b4ce2c013196 circuits=30200
# quantum: True | qubo vars: 630 | max qubits: 18
instance po_a010_t10_orig
budget 4
lambda 0.0001
objective -68249
# period symbol long short
0 AAPL 0 1
0 GOOGL 3 0
1 META 1 0
1 TSLA 0 3
2 META 3 0
2 NVDA 0 1
3 AVGO 0 1
3 GOOG 0 2
3 META 1 0
4 AMZN 1 0
4 META 2 0
4 TSLA 0 1
5 GOOGL 1 0
5 META 0 1
5 TSLA 2 0
6 GOOGL 1 0
6 MSFT 1 0
6 NVDA 0 1
6 TSLA 1 0
7 AVGO 0 1
7 NVDA 1 0
7 WMT 2 0
8 AAPL 3 0
8 AVGO 1 0
