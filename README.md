# Credit-Banking Attacks in Strategic Linear Contextual Bandits

The project studies whether a strategic arm can first under-report its context
to accumulate statistical "credit" and later over-report to gain additional
selections. It compares Per-arm LinUCB, one-sided OptGTM, bidirectional OptGTM,
and windowed bidirectional OptGTM in controlled and IID synthetic environments.

## Repository structure

```text
.
|-- experiments/                 # Single-condition simulation entry points
|-- scripts/
|   `-- run_credit_banking.py    # Complete reproducible experiment pipeline
|-- src/
|   |-- algorithms/              # LinUCB and OptGTM variants
|   |-- environment.py           # Controlled and IID environments
|   |-- metrics.py               # Episode and paired metrics
|   |-- plotting.py              # Figure generation
|   `-- strategies.py            # Truthful and strategic reporting policies
|-- tests/                       # Automated test suite
|-- results/
|   `-- credit_banking_experiment/ # Outputs used in the project report
`-- requirements.txt
```

## Requirements

- Python 3.11 or newer is recommended.
- The reported experiment was verified with Python 3.13.6.
- Required packages are listed in `requirements.txt`.

The verified environment used:

```text
numpy==2.5.4
matplotlib==3.11.2
pytest==9.2.0
```

The code does not require a GPU.

## Installation

Clone the repository and enter its directory:

```powershell
git clone https://github.com/DhanyasriChilaka/SCLB_proj_25210041.git
cd SCLB_proj_25210041
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the installation

Run the complete automated test suite from the repository root:

```powershell
python -m pytest -q
```

The version used for the report contains 140 tests. The expected final line is:

```text
140 passed
```

## Quick reproducibility check

Before running the full experiment, execute the smoke-sized protocol in a
separate output directory:

```powershell
python .\scripts\run_credit_banking.py `
  --quick `
  --stage all `
