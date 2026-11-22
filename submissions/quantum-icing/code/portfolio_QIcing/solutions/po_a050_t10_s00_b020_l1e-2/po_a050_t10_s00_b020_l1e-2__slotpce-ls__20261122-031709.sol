# QIcing portfolio attack | method: slot-PCE pools + period descent + LNS polish (hybrid)
# settings: slot formulation K=20 order=spectral, PCE multibasis2 n_layers=6 alpha=6.0 optimizer=lbfgs iters=200 repeats=8 pool=20; L1 period descent on 60 states/period; L2 lns 120s; circuits=30074; raw=-1000 L1=-1291; controls rand L2=-3537 empty L2=-1000
# quantum: True | qubo vars: 140/period | max qubits: 11
instance po_a050_t10_s00
budget 20
lambda 0.01
objective -2441
# period symbol long short
2 AAPL 0 1
2 ABT 1 0
2 CVX 1 0
2 GE 0 1
2 GOOG 1 0
2 GOOGL 0 1
2 LIN 0 1
2 MCD 0 1
2 MSFT 1 0
2 TMUS 0 1
2 V 1 0
