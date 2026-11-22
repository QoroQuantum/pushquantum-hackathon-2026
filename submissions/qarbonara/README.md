# Team Qarbonara — QOBLIB 06 portfolio submission

Team: Carles, Luca, Alberto, Mattia. Submitted by pull request because the aqora leaderboard page did not load
(as advised by the Qoro team on Slack, 22 Nov 10:22).

52 instances, one `.sol` (QOBLIB canonical format) + one `.json` (`qubits`, `method`) each. Every `.sol` passes
QOBLIB's official Rust checker (exact objective).

- 50 instances (10 and 50 assets): feasible-move blocks from cash; every stage partitioned and executed by Divi's
  PartitioningProgramEnsemble, one QAOA sub-program per block (p = 6, Characterization Service angles); all
  circuits on Qoro's cloud (qoro_cpu); exact dynamic program over days; scored with qoblib_tools.
  vs published BKV: 8 beat it (all eight 50-asset, λ = 1e-2 instances), 34 equal it, 8 within 0.1 %.
- 2 instances at 400 assets × 15 days (36 165 variables): po_a400_t15_orig λ 1e-2 from cash on Qoro's cloud
  (−2 757 451), and λ 1e-3 classical record refined by Divi QAOA on Qoro's cloud (−3 171 772).

`qubits` = sum of qubits across all partitions (as requested) where the per-partition sizes were recorded; otherwise
the largest single circuit, stated in that file's `method`.
