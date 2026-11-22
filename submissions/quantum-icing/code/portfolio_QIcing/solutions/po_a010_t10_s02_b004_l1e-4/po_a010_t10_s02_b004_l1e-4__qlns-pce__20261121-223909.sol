# QIcing portfolio attack | method: QLNS-pce (quantum sub-solver, Maestro)
# settings: routine=pce n_layers=3 window_assets=6 p_time=0.6 p_corr=0.3 p_hedge=0.0 hedge_assets=2 penalty_weight=5000 mc_pop=40 mc_iter=8 shots=4000 iters=1859 seed=16 start=/mnt/d/Repos/Quantum_Munich_Challenges_TUM_Hackathon_PushQuantum_2026/quantum-icing-qoro/portfolio_QIcing/solutions/po_a010_t10_s02_b004_l1e-4/po_a010_t10_s02_b004_l1e-4__qlns-pce__20261121-222342.sol warm_start=none char_job= circuits=596739
# quantum: True | qubo vars: 630 | max qubits: 9
instance po_a010_t10_s02
budget 4
lambda 0.0001
objective -53220
# period symbol long short
0 AAPL 0 1
0 AVGO 1 0
0 WMT 2 0
1 AVGO 0 1
1 MSFT 1 0
1 NVDA 1 0
1 WMT 0 1
2 MSFT 0 2
2 TSLA 2 0
3 AAPL 2 0
3 NVDA 1 0
3 TSLA 0 1
4 GOOG 3 0
4 TSLA 0 1
5 AVGO 0 1
5 GOOG 1 0
5 NVDA 1 0
5 TSLA 0 1
6 AAPL 0 2
6 AMZN 0 1
6 AVGO 1 0
7 AAPL 0 3
7 TSLA 1 0
8 AVGO 0 2
8 MSFT 0 1
8 NVDA 1 0
