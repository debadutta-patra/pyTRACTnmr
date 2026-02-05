import os
import numpy as np
import nmrglue as ng  # type: ignore
from scipy.optimize import curve_fit
from typing import Optional, Tuple, List, Dict
import logging
from typing_extensions import deprecated

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TractBruker:
    """
    Process Bruker TRACT NMR data for 15N relaxation analysis.
    """

    # Physical constants (CODATA 2018 values)
    PLANCK = 6.62607015e-34
    VACUUM_PERMEABILITY = 1.25663706212e-6
    GAMMA_1H = 267.52218744e6
    GAMMA_15N = -27.126e6
    NH_BOND_LENGTH = 1.02e-10
    CSA_15N = 160e-6
    CSA_BOND_ANGLE = 17 * np.pi / 180

    def __init__(self, exp_folder: str, delay_list: Optional[str] = None) -> None:
        """_summary_

        Args:
            exp_folder (str): Path to the Bruker experiment folder.
            delay_list (Optional[str], optional): Path to the delay list file. Defaults to None.

        Raises:
            ValueError: If the experiment cannot be loaded.
        """
        logger.info(f"Initializing TractBruker with folder: {exp_folder}")

        try:
            self.attributes, self.fids = ng.bruker.read(exp_folder)
            try:
                self.phc0 = self.attributes["procs"]["PHC0"]
                self.phc1 = self.attributes["procs"]["PHC1"]
            except KeyError:
                self.phc0 = 0.0
                self.phc1 = 0.0
        except Exception as e:
            raise ValueError(f"Could not load Bruker data: {e}")

        # Handle delays
        if delay_list and os.path.exists(delay_list):
            self.delays = self._read_delays(delay_list)
        else:
            # Try standard 'vdlist' in folder
            vdlist_path = os.path.join(exp_folder, "vdlist")
            if os.path.exists(vdlist_path):
                self.delays = self._read_delays(vdlist_path)
            else:
                raise ValueError("No delay list found (vdlist) and no external list provided.")

        self.alpha_spectra: List[np.ndarray] = []
        self.beta_spectra: List[np.ndarray] = []
        # self.alpha_integrals: np.ndarray | None = None
        # self.beta_integrals: np.ndarray | None = None
        self.unit_converter = None

    def _read_delays(self, file: str) -> np.ndarray:
        """Uitility function for reading vdlist file and converting it to numpy ndarray

        Args:
            file (str): Path to the vdlist file

        Returns:
            np.ndarray: Numpy array condaining the delays in seconds.
        """
        with open(file, "r") as list_file:
            delays = list_file.read()
        delays = delays.replace("u", "e-6").replace("m", "e-3")
        return np.array([float(x) for x in delays.splitlines() if x.strip()])

    def _get_lb_val(self, lb: float) -> float:
        """Calculate normalized line broadening value."""
        try:
            sw = self.attributes["acqus"]["SW_h"]
            return lb / sw
        except KeyError, ZeroDivisionError:
            return lb

    def _process_single_fid(
        self, fid, p0, p1, points, apod_func, lb_val, off, end, pow, nodes
    ) -> np.ndarray:
        """Internal helper to process a single FID."""
        # Apply apodization
        if apod_func == "em":
            data = ng.proc_base.em(fid, lb=lb_val)
        else:
            data = ng.proc_base.sp(fid, off=off, end=end, pow=pow)

        # Zero filling
        data = ng.proc_base.zf_size(data, points)
        # Fourier transform
        data = ng.proc_base.fft(data)
        # Remove digital filter
        data = ng.bruker.remove_digital_filter(self.attributes, data, post_proc=True)
        # Apply phase correction
        data = ng.proc_base.ps(data, p0=p0, p1=p1)
        # Discard imaginary part
        data = ng.proc_base.di(data)
        # Reverse spectrum
        data = ng.proc_base.rev(data)
        if nodes is not None and len(nodes) > 1:
            data = ng.proc_bl.base(data, nodes)
        return data

    def process_first_trace(
        self,
        p0: float,
        p1: float,
        points: int = 2048,
        apod_func: str = "sp",
        lb: float = 0.0,
        off: float = 0.35,
        end: float = 0.98,
        pow: float = 2.0,
        nodes=None,
    ) -> np.ndarray:
        """Process the first plane in the Psuedo-2D experiment. This is useful for phase correction

        Args:
            p0 (float): Zeroth order phase correction
            p1 (float): First order phase correction
            points (int, optional): Zero filling points. Defaults to 2048.
            apod_func (str, optional): Apodization function to use. Only "sp" and "em" are supported. Defaults to "sp".
            lb (float, optional): Line broadening in Hz (only for em). Defaults to 0.0.
            off (float, optional): Offset for sp apodization. Defaults to 0.35.
            end (float, optional): End of sp apodization. Defaults to 0.98.
            pow (float, optional): Power for sp apodization. Defaults to 2.0.

        Returns:
            np.ndarray: Fourier transformed spectrum containng only the real part.
        """
        fid = self.fids[0, 0]
        lb_val = self._get_lb_val(lb) if apod_func == "em" else 0.0
        data = self._process_single_fid(
            fid, p0, p1, points, apod_func, lb_val, off, end, pow, nodes
        )

        # Set up unit converter
        udic = ng.bruker.guess_udic(self.attributes, data)
        self.unit_converter = ng.fileiobase.uc_from_udic(udic)
        return data

    def split_process(
        self,
        p0: float,
        p1: float,
        points: int = 2048,
        apod_func: str = "sp",
        lb: float = 0.0,
        off: float = 0.35,
        end: float = 0.98,
        pow: float = 2.0,
        nodes=None,
    ) -> None:
        """The primary function for processing the Pusedo-2D experiment. This splits the data into alpha and beta state and perform the basic processing of the raw FIDs

        Args:
            p0 (float): Zeroth order phase correction.
            p1 (float): First order phase correction.
            points (int, optional): Zero filling points. Defaults to 2048.
            apod_func (str, optional): Apodization function to use. Only "sp" and "em" are supported. Defaults to "sp".
            lb (float, optional): Line broadening in Hz (only for em). Defaults to 0.0.
            off (float, optional): Offset for sp apodization. Defaults to 0.35.
            end (float, optional): End of sp apodization. Defaults to 0.98.
            pow (float, optional): Power for sp apodization. Defaults to 2.0.
        """
        self.phc0 = p0
        self.phc1 = p1
        self.alpha_spectra = []
        self.beta_spectra = []

        lb_val = self._get_lb_val(lb) if apod_func == "em" else 0.0

        for i in range(self.fids.shape[0]):
            for j in range(self.fids[i].shape[0]):
                data = self._process_single_fid(
                    self.fids[i][j],
                    p0,
                    p1,
                    points,
                    apod_func,
                    lb_val,
                    off,
                    end,
                    pow,
                    nodes,
                )
                if j % 2 == 0:
                    self.beta_spectra.append(data)
                else:
                    self.alpha_spectra.append(data)

        # Unit converter from first spectrum
        if self.beta_spectra:
            udic = ng.bruker.guess_udic(self.attributes, self.beta_spectra[0])
            self.unit_converter = ng.fileiobase.uc_from_udic(udic)

    @deprecated("Use integrate_ppm() instead")
    def integrate_indices(self, start_idx: int, end_idx: int) -> None:
        """Intergrate the specified region in the spectra. This accounts for all the alpha and beta spectrum collected.

        Args:
            start_idx (int): Start index for integration.
            end_idx (int): End index for integration.


        Raises:
            RuntimeError: If no spectra are available. Run split_process() first.
        """
        if not self.alpha_spectra or not self.beta_spectra:
            raise RuntimeError("No spectra available. Run split_process() first.")

        self.alpha_integrals: np.ndarray = np.array(
            [s[start_idx:end_idx].sum() for s in self.alpha_spectra]
        )
        self.beta_integrals: np.ndarray = np.array(
            [s[start_idx:end_idx].sum() for s in self.beta_spectra]
        )

    def integrate_ppm(self, start_ppm: float, end_ppm: float) -> None:
        """Integrate the specified region in all the extracted spectras.

        Args:
            start_ppm (float): Start index for integration in ppm.
            end_ppm (float): End index for integration in ppm.

        Raises:
            RuntimeError: If no spectra are available. Run split_process() first.
            RuntimeError: If no unit converter is available. Run split_process() first.
        """
        if self.unit_converter is None:
            raise RuntimeError("Unit converter not initialized.")

        idx1 = self.unit_converter(start_ppm, "ppm")
        idx2 = self.unit_converter(end_ppm, "ppm")

        start = int(min(idx1, idx2))
        end = int(max(idx1, idx2))
        self.integrate_indices(start, end)

    @staticmethod
    def _relax(x: np.ndarray[np.float64], a: float, r: float) -> np.ndarray[np.float64]:
        """Internal function for exponential decay

        Args:
            x (np.ndarray): X values
            a (float): Amplitude
            r (float): Decay rate

        Returns:
            float | np.ndarray: Y values
        """
        return a * np.exp(-r * x)

    def calc_relaxation(self) -> None:
        """Calculate the Relaxation rates for alpha and beta states. This function does not return any values but sets

        Raises:
            RuntimeError: If no integrals are available. Run integrate_ppm() first.
            RuntimeError: If fitting fails.
        """
        if self.alpha_integrals is None or self.beta_integrals is None:
            raise RuntimeError("Must call integrate() before calc_relaxation()")

        # Truncate delays if mismatch
        n_pts = min(len(self.alpha_integrals), len(self.delays))
        delays: np.ndarray = self.delays[:n_pts]
        alpha_ints = self.alpha_integrals[:n_pts]
        beta_ints = self.beta_integrals[:n_pts]

        # Normalize
        alpha_norm = alpha_ints / alpha_ints[0]
        beta_norm = beta_ints / beta_ints[0]

        try:
            self.popt_alpha, self.pcov_alpha = curve_fit(
                self._relax, delays, alpha_norm, p0=[1.0, 5.0], maxfev=5000
            )
            self.popt_beta, self.pcov_beta = curve_fit(
                self._relax, delays, beta_norm, p0=[1.0, 5.0], maxfev=5000
            )
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {e}")

        self.Ra: float = self.popt_alpha[1]
        self.Rb: float = self.popt_beta[1]
        self.err_Ra: float = np.sqrt(np.diag(self.pcov_alpha))[1]
        self.err_Rb: float = np.sqrt(np.diag(self.pcov_beta))[1]

    def _tc_equation(self, w_N: float, c: float, S2: float = 1.0) -> float:
        """Function for calculating the Rotational Correlation Time. The equation is are adapted from eq. 15 of:
        'TRACT revisited: an algebraic solution for determining overall rotational correlation times from cross-correlated relaxation rates'
        PMID: 34480265
        doi: 10.1007/s10858-021-00379-5

        Args:
            w_N (float): Larmor Frequency of Nitrogen atom.
            c (float): Constant derived from the relaxation rate of the alpha state and beta state.
            S2 (float, optional): Square of the order parameter. Defaults to 1.0.

        Returns:
            float: Rotational Correlation Time in ns.
        """
        t1 = (5 * c) / (24 * S2)
        A = 336 * (S2**2) * (w_N**2)
        B = 25 * (c**2) * (w_N**4)
        C = 125 * (c**3) * (w_N**6)
        D = 625 * (S2**2) * (c**4) * (w_N**10)
        E = 3025 * (S2**4) * (c**2) * (w_N**8)
        F = 21952 * (S2**6) * (w_N**6)
        G = 1800 * c * (w_N**4)
        term_sqrt = np.sqrt(D - E + F)
        term_cbrt = (C + 24 * np.sqrt(3) * term_sqrt + G * S2**2) ** (1 / 3)
        t2 = (A - B) / (24 * (w_N**2) * S2 * term_cbrt)
        t3 = term_cbrt / (24 * S2 * w_N**2)
        return t1 - t2 + t3

    def calc_tc(
        self, B0: Optional[float] = None, S2: float = 1.0, n_bootstrap: int = 1000
    ) -> None:
        """Calculate Rotational Correlation Time by using bootstraping. The Relaxation rates are resampled based on the error estimates derived from the covaraince matrix.

        Args:
            B0 (Optional[float], optional): Magnetic field in MHz. Defaults to None.
            S2 (float, optional): Square of the Order parameter. Defaults to 1.0.
            n_bootstrap (int, optional): Number of bootstrap samples. Defaults to 1000.
        """
        if not hasattr(self, "Ra"):
            self.calc_relaxation()
        if B0 is None:
            B0 = self.attributes["acqus"]["SFO1"]
        B_0 = B0 * 1e6 * 2 * np.pi / self.GAMMA_1H
        p = (
            self.VACUUM_PERMEABILITY * self.GAMMA_1H * self.GAMMA_15N * self.PLANCK
        ) / (16 * np.pi**2 * np.sqrt(2) * self.NH_BOND_LENGTH**3)
        dN = self.GAMMA_15N * B_0 * self.CSA_15N / (3 * np.sqrt(2))
        w_N = B_0 * self.GAMMA_15N
        Ra_samples: np.ndarray = np.random.normal(self.Ra, self.err_Ra, n_bootstrap)
        Rb_samples: np.ndarray = np.random.normal(self.Rb, self.err_Rb, n_bootstrap)
        c_samples = (Rb_samples - Ra_samples) / (
            2 * dN * p * (3 * np.cos(self.CSA_BOND_ANGLE) ** 2 - 1)
        )
        tau_samples: np.ndarray = (
            np.array(
                [self._tc_equation(w_N, c, S2) for c in c_samples if not np.isnan(c)]
            )
            * 1e9
        )
        self.tau_c = np.mean(tau_samples)
        self.err_tau_c = np.std(tau_samples)

    def calc_confidence_interval(
        self, x: np.ndarray, popt: np.ndarray, pcov: np.ndarray
    ) -> np.ndarray:
        """Calculate 95% confidence interval for the exponential decay."""
        A, R = popt
        # Gradient of f(x) = A * exp(-R * x)
        # df/dA = exp(-R * x)
        # df/dR = -A * x * exp(-R * x)
        df_dA = np.exp(-R * x)
        df_dR = -A * x * np.exp(-R * x)

        J = np.stack([df_dA, df_dR], axis=1)

        # sigma^2 = diag(J @ pcov @ J.T)
        sigma2 = np.sum((J @ pcov) * J, axis=1)
        return 1.96 * np.sqrt(sigma2)

    def get_fit_data(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Returns the fit data for alpha and beta states

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (Delays (s), Ratios of alpha state, Ratios of beta state, optimized parameters for alpha state, optimized parameters for beta state, cov matrix alpha, cov matrix beta)
        """
        n_pts = min(len(self.alpha_integrals), len(self.delays))
        x = self.delays[:n_pts]
        y_a = self.alpha_integrals[:n_pts] / self.alpha_integrals[0]
        y_b = self.beta_integrals[:n_pts] / self.beta_integrals[0]
        return (
            x,
            y_a,
            y_b,
            self.popt_alpha,
            self.popt_beta,
            self.pcov_alpha,
            self.pcov_beta,
        )
