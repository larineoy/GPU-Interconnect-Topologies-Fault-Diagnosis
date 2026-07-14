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
| 5 | Observation matrix (on-the-fly) | Done |
| 6 | Information-gain algorithm | Done |
| 7 | Simulator | Done |
| 8 | Main experiments | Done |
| 9 | Ablation studies | Done |
| 10 | Paper figures | Done |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_topology.py   # if topology JSON is missing
```

## Run main experiment

```bash
python experiments/run_main.py                  # 1000 trials (default)
python experiments/run_main.py --trials 100     # smoke run
```

Outputs land in `results/` (`main_trials.csv`, `main_summary.json`).

## Run ablations

```bash
python experiments/run_ablation.py --trials 500
```

Compares: InfoSlice, random selection, no duration weighting, uniform priors, NVL36 vs NVL72 scale.

## Generate paper figures

```bash
python experiments/generate_figures.py
```

Writes PNGs to `paper/figures/` (and `results/figures/`).

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
