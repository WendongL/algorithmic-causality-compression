# Algorithmic causal structure emerging through compression

Paper: https://arxiv.org/abs/2502.04210

This repository contains the code for verifying our paper on algorithmic causality using synthetic data. The experiments include causal discovery and covariate shift.

## Project Structure

```
algorithmic-causality-compression/
├── causal_discovery/ # Causal discovery scripts
├── covariate_shift/  # Covariate shift scripts
├── configs/              # Configuration files
├── utils.py            # Utility functions
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── LICENSE               # License information
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WendongL/algorithmic-causality-compression.git
   cd algorithmic-causality-compression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Causal Discovery
Run causal discovery experiments:
```bash
python causal_discovery/main.py --settings_file configs/example_causal_discovery.yaml
```

### Covariate Shift
Run covariate shift experiments:
```bash
python covariate_shift/main.py --settings_file configs/example_covariate_shift.yaml
```

### Plotting
Generate plots for results:
```bash
python causal_discovery/plot.py --config ../[your output folder]/[one of your yaml files in the output folder].yaml
python covariate_shift/plot.py --config ../[your output folder]/[one of your yaml files in the output folder].yaml
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.