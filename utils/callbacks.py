from typing import Any, Dict, Optional


class EarlyStopper:
    """
    Early stopping utility for loss-based validation monitoring.

    The stopper tracks the best (lowest) validation loss seen so far.
    Training is stopped once there have been `patience` consecutive epochs
    without a *meaningful* improvement, where "meaningful" means the loss
    decreased by more than `min_delta`.

    Notes
    -----
    - An epoch counts as an improvement **only if**:
        current_loss <= best_loss - min_delta
      Tiny fluctuations smaller than or equal to `min_delta` are ignored.
    - If the loss does not improve meaningfully, the internal counter increases.
      Once it reaches `patience`, `early_stop(...)` returns True.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0.0) -> None:
        """
        Initialize the EarlyStopper.

        Parameters
        ----------
        patience : int, optional
            Number of consecutive epochs with no meaningful improvement after
            which training will be stopped. Must be >= 1. Default is 1.
        min_delta : float, optional
            Minimum absolute decrease in the monitored loss to qualify as a
            meaningful improvement. Must be >= 0.0. Default is 0.0.

        Raises
        ------
        ValueError
            If `patience < 1` or `min_delta < 0`.
        """
        if patience < 1:
            raise ValueError("`patience` must be >= 1.")
        if min_delta < 0:
            raise ValueError("`min_delta` must be >= 0.")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_results: Optional[Dict[str, Any]] = None
        self.best_model_state: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_float(x: Any) -> float:
        """Safely convert a float or a tensor-like object with `.item()` to float."""
        if hasattr(x, "item") and callable(getattr(x, "item")):
            return float(x.item())
        return float(x)

    def early_stop(self, validation_results: Dict[str, Any], model_state: Dict[str, Any]) -> bool:
        """
        Update stopper with the latest validation results and decide whether to stop.

        Parameters
        ----------
        validation_results : dict
            Dictionary containing validation metrics. Must include key ``"loss"`` whose
            value is a float or tensor-like with ``.item()``.
        model_state : dict
            Model state corresponding to the current epoch/bucket (e.g., a checkpoint
            state dict or any serializable snapshot).

        Returns
        -------
        bool
            True if early stopping criterion is met (i.e., no meaningful improvement
            for `patience` consecutive checks), False otherwise.

        Raises
        ------
        KeyError
            If ``"loss"`` is not present in `validation_results`.
        """
        if "loss" not in validation_results:
            raise KeyError('`validation_results` must contain key "loss".')

        validation_loss = self._to_float(validation_results["loss"])

        # Determine if there is a meaningful improvement.
        # Improvement if current <= best - min_delta
        if validation_loss <= (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_results = validation_results
            self.best_model_state = model_state
            self.counter = 0
            return False  # improved -> definitely do not stop
        else:
            # No meaningful improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def get_best_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the best (lowest-loss) validation results observed.

        Returns
        -------
        dict or None
            Dictionary with the best validation results, or None if no update has occurred yet.
        """
        return self.best_results

    def get_best_model_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the model state corresponding to the best validation results.

        Returns
        -------
        dict or None
            Dictionary with the best model state, or None if no update has occurred yet.
        """
        return self.best_model_state
