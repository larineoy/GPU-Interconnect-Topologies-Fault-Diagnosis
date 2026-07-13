# InfoSlice

Information-theoretic fault diagnosis for NVIDIA NVL72 GPU interconnect fabrics.

When a burn-in test fails on an NVL72 UltraServer (72 GPUs connected via 18 NVSwitches), InfoSlice selects diagnostic tests by maximizing mutual information per unit time to identify the faulty component (GPU, NVSwitch, NVLink, or compute tray).

## Status

Repository scaffolding is in place. Implementation proceeds by step:

| Step | Description | Status |
|------|-------------|--------|
| 1 | Repository setup | Done |
| 2 | Topology generation + graph loader | Done |
| 3 | Failure priors + hypothesis space | Done |
| 4 | Test catalog | Done |
| 5 | Observation matrix (on-the-fly) | Pending |
| 6 | Information-gain algorithm | Pending |
| 7 | Simulator | Pending |
| 8 | Main experiments | Pending |
| 9 | Ablation studies | Pending |
| 10 | Paper figures | Pending |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
infoslice/
├── scripts/generate_topology.py   # Generate NVL72 topology JSON
├── data/                          # Public-source topology, priors, tests
├── src/                           # Core library
│   ├── topology/                  # NetworkX graph
│   ├── failure_model/             # Hypotheses + priors
│   ├── test_model/                # Catalog + observation outcomes
│   ├── algorithm/                 # MI, greedy selection, Bayesian update
│   ├── simulator/                 # Fault injection + experiment runner
│   └── utils/                     # Metrics
├── experiments/                   # Main + ablation entry points
├── results/                       # Generated outputs (gitignored)
├── notebooks/                     # Exploration
└── tests/                         # Unit tests
```

## Data policy

All topology, failure, and test data must come from **public sources** (NVIDIA docs, published papers). No proprietary or internal data.

## License

See [LICENSE](LICENSE).
