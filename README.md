# pyTRACTnmr

**pyTRACTnmr** is a graphical user interface (GUI) application designed for the processing and analysis of TRACT (TROSY for Rotational Correlation Times) experiments in NMR spectroscopy. It provides a streamlined workflow to go from raw Bruker data to calculated rotational correlation times ($\tau_c$) with robust error estimation. Currently this only supports collected with Bruker spectrometers with pulseprogram `tractf3gpphwg`.

## Features

- **User-Friendly Interface**: Built with PySide6 for a responsive and native experience.
- **Bruker Data Import**: Directly load Bruker experiment directories.
- **Interactive Spectral Processing**:
  - Real-time 0th and 1st order phase correction.
  - Adjustable Apodization (Sine-bell squared) and Zero Filling.
  - **Interactive Region Selection**: Drag directly on the spectrum to define integration limits.
- **Advanced Analysis**:
  - Automatic calculation of relaxation rates ($R_\alpha$ and $R_\beta$).
  - Determination of Rotational Correlation Time ($\tau_c$).
  - **Bootstrap Error Analysis**: rigorous uncertainty estimation for $\tau_c$.
- **Data Management**:
  - Tabular view of multiple loaded experiments.
  - Context menu to export results to CSV.

## Installation

### Prerequisites

*   Python 3.14 or higher recommended (Though it is not tested it should run with earlier version of Python 3).
*   [uv](https://github.com/astral-sh/uv) (optional, but recommended for building).

### Installation 
#### Quick Start with uv
The fastest way to try pyTRACTnmr without installation is:

```bash
uvx pytractnmr
```

#### Using pip

'''bash
pip install pyTRACTnmr
'''

#### From Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/debadutta-patra/pyTRACTnmr.git
    cd pyTRACTnmr
    ```

2.  **Install the package:**

    Using `uv` (fastest):
    ```bash
    uv pip install .
    ```

    Using standard `pip`:
    ```bash
    pip install .
    ```

## Usage

### Launching the App

After installation, you can start the application from the terminal:

```bash
pytractnmr
```

Or run it as a python module:

```bash
python -m pyTRACTnmr.main
```

### Analysis Workflow

1.  **Load Data**: Click **"Load Bruker Directory"** and select your experiment folder.
2.  **Process**:
    *   Use the **Processing** tab to adjust phase correction sliders.
    *   Drag the **green selection box** on the top spectrum plot to define the integration region (Start/End ppm).
3.  **Fit**:
    *   Switch to the **Fitting** tab.
    *   Input experimental parameters (Field Strength, CSA, etc.).
    *   Set the number of **Bootstraps** (e.g., 1000).
    *   Click **"Calculate Tau_c"**.
4.  **Export**:
    *   Right-click the results table to **Export Table to CSV**.

## Dependencies

*   `PySide6`
*   `numpy`
*   `scipy`
*   `matplotlib`
*   `nmrglue`

## License

This project is licensed under the MIT License - see the LICENSE file for details.