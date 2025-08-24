# Thermal Physical Learning in Sparse Networks (TPLISN)

## Overview

This repository contains the implementation and experimental code for the Master's thesis "Thermal Physical Learning in Sparse Networks" by Thomas Pluck, exploring whether physical learning systems can be understood through generalized Ising dynamics with finite-time field relaxation.

## Abstract

Recent physical learning systems have demonstrated that networks can learn computational tasks through local adaptation without central control, but the relationship between different physical learning approaches remains unclear. This work explores whether such systems can be understood through generalized Ising dynamics with finite-time field relaxation. We model spin expectations as beta-distributed random variables over finite time horizons, capturing uncertainty in non-equilibrium sampling while maintaining analytical tractability. Rather than requiring equilibrium spin correlations, we develop a field-discrepancy learning rule that operates on local voltage differences, similar to adaptation mechanisms observed in physical resistor networks. Experimental validation using logical routing tasks on sparse lattices demonstrates that the framework can reproduce learning behavior with only nearest-neighbor connectivity.

## Key Contributions

1. **Theoretical Framework**: Connects discrete Ising dynamics with continuous physical learning through beta-distributed spin expectations and finite field relaxation times
2. **Field-Discrepancy Learning**: A learning algorithm that operates on local voltage differences without requiring equilibrium states  
3. **Sparse Network Learning**: Experimental demonstration that sparse 2D lattice connectivity can support nonlinear learning tasks

## Repository Structure

```
├── beta_arch.py              # Beta-distributed Ising model implementation
├── clln_arch.py              # Classical Ising node-edge architecture
├── data.py                   # XOR training data definitions
├── debug_tools.py            # Visualization utilities for debugging
├── parallel_experiments.py   # Single experiment runner
├── sweep.py                  # Parameter sweep experiment framework
├── visualize_sweep_results.py # Analysis and visualization of sweep results
└── README.md                 # This file
```

## Core Architecture

### Beta-Distributed Ising Model (`beta_arch.py`)

The core innovation is modeling spin expectations as beta-distributed random variables:

```python
ŝᵢᵗ⁺¹ ~ 2 · Beta(τ σ(βhᵢᵗ), τ (1 - σ(βhᵢᵗ))) - 1
```

Where:
- `τ` is the finite relaxation time parameter
- `β` is the inverse temperature
- `hᵢᵗ` is the local magnetic field at time t
- `σ(x) = (1 + tanh(x))/2` is the sigmoid function

### Field-Discrepancy Learning

The learning rule updates connection weights based on squared field differences between free and clamped network states:

```python
ΔJᵢⱼ ∝ [Δκᶠʳᵉᵉ]² - [Δκᶜˡᵃᵐᵖᵉᵈ]²
```

This approach:
- Operates on local voltage differences
- Doesn't require equilibrium states
- Mimics adaptation mechanisms in physical resistor networks

## Experimental Setup

### XOR Learning Task

The framework is validated on XOR logical routing using a 4×4 sparse lattice:

- **Input Encoding**: Corner nodes clamped to ±1 values (bipolar encoding: 0→-1, 1→+1)
- **Output Reading**: Computed as difference between designated output nodes
- **Sparse Connectivity**: Only nearest-neighbor connections in 2D grid
- **Learning Protocol**: Contrastive learning comparing free vs clamped network responses

### Parameter Studies

Comprehensive parameter sweeps across:
- **Relaxation times (τ)**: 5, 10, 20, 50, 100
- **Learning rates (α)**: 10⁻⁵, 10⁻⁴, 10⁻³  
- **Nudging strengths (η)**: 0.1, 0.25, 0.5, 1.0
- **Runs per configuration**: 10 independent trials
- **Training trials**: 500 trials per run

## Usage

### Running Single Experiments

```bash
python parallel_experiments.py
```

This runs a single experiment configuration and generates:
- Training accuracy plots
- Final network state visualization
- Connection weight visualizations

### Parameter Sweep

```bash
python sweep.py
```

Runs comprehensive parameter sweeps across all configurations with:
- Parallel execution across multiple CPU cores
- Automatic checkpointing and resume capability
- Real-time progress monitoring
- Error handling and analytics

Results are saved in `parameter_sweep_results/` directory.

### Analyzing Results

```bash
python visualize_sweep_results.py
```

Generates comprehensive analysis including:
- Parameter performance heatmaps
- Convergence analysis
- Statistical comparisons
- Best/worst configuration analysis

## Key Findings

The experimental validation demonstrates:

1. **Learning Capability**: Networks can learn XOR classification without explicit parameter programming
2. **Sparse Connectivity**: Nearest-neighbor connections sufficient for nonlinear learning tasks
3. **Parameter Sensitivity**: Performance varies significantly across different τ, α, and η values
4. **Robustness**: Framework maintains reasonable performance across multiple operating regimes

## Installation

```bash
git clone https://github.com/ThomasPluck/TPLISN.git
cd TPLISN
pip install -r requirements.txt
```

## Technical Details

### Mathematical Framework

The system models physical learning networks as stochastic spin systems where:

1. **Node States**: Evolve according to local magnetic fields with finite relaxation times
2. **Edge States**: Beta-distributed spin expectations capture uncertainty in finite-time sampling  
3. **Weight Updates**: Based on field discrepancies rather than spin correlations
4. **Sparse Topology**: 2D lattice with antisymmetric couplings preserving current conservation

### Comparison with Traditional Approaches

| Framework | Control | Computation | Adaptation |
|-----------|---------|-------------|------------|
| Traditional | Explicit | Deterministic | Programmed |
| Probabilistic | Explicit | Equilibrium | Optimization |
| **This Work** | **Implicit** | **Fluctuation** | **Self-Organization** |

## Future Directions

The framework opens several research directions:

1. **Scalability Studies**: Larger network sizes and more complex tasks
2. **Hardware Implementation**: Physics-based ASIC design implications
3. **Alternative Topologies**: Beyond sparse 2D lattices
4. **Task Diversity**: More complex computational primitives
5. **Energy Efficiency**: Comparison with digital approaches

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{pluck2025thermal,
  title={Thermal Physical Learning in Sparse Networks},
  author={Pluck, Thomas},
  year={2025},
  school={Maynooth University},
  type={Master's Thesis}
}
```

## License

This work is provided for academic and research purposes. Please see the full thesis for detailed licensing information.

## Acknowledgments

Grateful acknowledgment to:
- Gerry Lacey and Majid Sorouri for research guidance
- John Dooley, Barak Pearlmutter, and Zachary Belateche for technical support
- Kerem Camsari, Aida Todri-Sanial, Todd Hylton, and Philippe Talatchian for field insights
- Joana and family for support throughout this work

## Contact

For questions about this research or collaboration opportunities, please contact Thomas Pluck through the GitHub repository issues or Maynooth University.
