# QIcing portfolio attack | method: QLNS-pce (quantum sub-solver, Maestro)
# settings: routine=pce n_layers=3 window_assets=6 p_time=0.6 p_corr=0.3 p_hedge=0.0 hedge_assets=2 penalty_weight=5000 mc_pop=40 mc_iter=8 shots=4000 iters=1937 seed=15 start=/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/portfolio_QIcing/solutions/po_a010_t10_orig_b004_l1e-3/po_a010_t10_orig_b004_l1e-3__qlns-pce__20261121-222545.sol warm_start=none char_job= circuits=621777
# quantum: True | qubo vars: 630 | max qubits: 9
instance po_a010_t10_orig
budget 4
lambda 0.001
objective -8397
# period symbol long short
0 AAPL 0 1
0 GOOGL 1 0
1 META 1 0
1 MSFT 0 1
1 WMT 0 1
2 AMZN 1 0
2 MSFT 0 1
3 AMZN 0 1
3 GOOGL 0 1
3 META 1 0
3 WMT 0 1
4 AMZN 1 0
5 AAPL 0 1
5 AMZN 1 0
5 META 0 1
5 TSLA 1 0
7 AMZN 1 0
7 MSFT 0 1
7 WMT 1 0
8 AAPL 2 0
8 META 0 1
8 NVDA 1 0
