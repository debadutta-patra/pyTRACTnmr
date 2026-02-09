import copy
from PySide6.QtCore import QObject, Signal

class SlidingWindowWorker(QObject):
    finished = Signal(object, object, object, int)
    error = Signal(str)

    def __init__(self, tb, start_ppm, end_ppm, b0, s2, n_boot, dataset_idx):
        super().__init__()
        self.tb = tb
        self.start_ppm = start_ppm
        self.end_ppm = end_ppm
        self.b0 = b0
        self.s2 = s2
        self.n_boot = n_boot
        self.dataset_idx = dataset_idx

    def run(self):
        try:
            tb_copy = copy.copy(self.tb)
            ppms, taus, errs = tb_copy.calc_sliding_window(
                self.start_ppm,
                self.end_ppm,
                window_pct=0.05,
                B0=self.b0,
                S2=self.s2,
                n_bootstrap=self.n_boot,
            )
            self.finished.emit(ppms, taus, errs, self.dataset_idx)
        except Exception as e:
            self.error.emit(str(e))
