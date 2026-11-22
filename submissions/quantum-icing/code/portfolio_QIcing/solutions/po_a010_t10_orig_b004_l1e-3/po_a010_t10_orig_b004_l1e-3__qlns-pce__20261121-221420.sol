# QIcing portfolio attack | method: QLNS-pce (quantum sub-solver, Maestro)
# settings: routine=pce n_layers=3 window_assets=4 p_time=0.4 penalty_weight=5000 mc_pop=40 mc_iter=8 shots=4000 iters=400 seed=6 start=empty warm_start=none char_job= circuits=128400
# quantum: True | qubo vars: 630 | max qubits: 8
instance po_a010_t10_orig
budget 4
lambda 0.001
objective -6962
# period symbol long short
2 AMZN 1 0
2 MSFT 0 1
4 AMZN 1 0
7 AMZN 1 0
7 MSFT 0 1
7 WMT 1 0
8 AAPL 2 0
8 META 0 1
8 NVDA 1 0
