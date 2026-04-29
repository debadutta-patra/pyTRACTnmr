# pyTRACTnmr

**pyTRACTnmr** is a graphical user interface (GUI) application designed for the processing and analysis of TRACT (TROSY for Rotational Correlation Times) experiments in NMR spectroscopy. It provides a streamlined workflow to go from raw Bruker data to calculated rotational correlation times ($\tau_c$) with robust error estimation. All calculations are based on **TRACT revisited: an algebraic solution for determining overall rotational correlation times from cross-correlated relaxation rates** [DOI:10.1007/s10858-021-00379-5](https://doi.org/10.1007/s10858-021-00379-5). 

**Note:** Currently this only supports data collected with Bruker spectrometers using pulseprogram `tractf3gpphwg`.

## Features

*   **Bruker Data Import**: Directly load Bruker experiment directories.
*   **Interactive Processing**:
    *   Phase correction.
    *   Apodization (Sine Bell, Exponential) and Zero Filling.
    *   Baseline correction with manual node picking.
*   **Relaxation Analysis**:
    *   Automatic splitting of Pseudo-2D experiments into $\alpha$ (Anti-TROSY) and $\beta$ (TROSY) states.
    *   Exponential decay fitting to determine relaxation rates ($R_\alpha$, $R_\beta$).
    *   Calculation of $\tau_c$ with error estimation using bootstrapping.
*   **Sliding Window Analysis**: Perform $\tau_c$ calculation across the spectral width to identify domain dynamics.
*   **Visualization**: Interactive Matplotlib plots for spectra, decay fits, and sliding window results embedded in a Qt interface.
*   **Export**: Save results to CSV and generate Python scripts for publication-quality plotting.

## Installation

### Prerequisites

*   Python 3.11 or higher recommended.
*   [uv](https://github.com/astral-sh/uv) (optional, but recommended for building).

### Installation 
#### Using uv
The fastest way to try pyTRACTnmr without installation is:

```bash
uvx pyTRACTnmr
```
To install:
```bash
uv tool install pyTRACTnmr
```
#### Using pip

```bash
pip install pyTRACTnmr
```

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
    *   Select nodes for polynomial baseline correction using the **Pick Nodes** button.
3.  **Fit**:
    *   Switch to the **Fitting** tab.
    *   Input experimental parameters (Field Strength, CSA, etc.).
    *   Set the number of **Bootstraps** (e.g., 10000).
    *   Click **"Calculate Tau_c"**.
    *   (Optional) Check the sliding window analysis to calculate $\tau_c$ across the spectral width.
4.  **Export**:
    *   Right-click the results table to **Export Table to CSV**.
    *   Click the export options in **Fitting** tab to generate a CSV file and python script for publication-quality plotting.
    *   A PDF report with all generated plots and processing parameters can be exported.


## Dependencies

*   `PySide6`
*   `numpy`
*   `scipy`
*   `matplotlib`
*   `nmrglue`

## License

This project is licensed under the GNU General Public License v3.0 License - see the LICENSE file for details.
