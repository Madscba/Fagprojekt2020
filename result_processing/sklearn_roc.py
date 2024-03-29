from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.utils import check_matplotlib_support


class RocCurveDisplay:
    """ROC Curve visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
    visualizer. All parameters are stored as attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    fpr : ndarray
        False positive rate.
    tpr : ndarray
        True positive rate.
    roc_auc : float, default=None
        Area under ROC curve. If None, the roc_auc score is not shown.
    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.
    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.
    ax_ : matplotlib Axes
        Axes with ROC Curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    >>> roc_auc = metrics.auc(fpr, tpr)
    >>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,\
                                          estimator_name='example estimator')
    >>> display.plot()  # doctest: +SKIP
    >>> plt.show()      # doctest: +SKIP
    """
    def __init__(self, *, fpr, tpr, roc_auc=None, estimator_name=None):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_name = estimator_name

    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization
        Extra keyword arguments will be passed to matplotlib's ``plot``.
        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.
        Returns
        -------
        """
        check_matplotlib_support('RocCurveDisplay.plot')
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs["label"] = f"{name} (AUC = {self.roc_auc:0.2f})"
        elif self.roc_auc is not None:
            line_kwargs["label"] = f"AUC = {self.roc_auc:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        if "label" in line_kwargs:
            ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

def plot_roc_curve(y, y_pred, sample_weight=None,
                   drop_intermediate=True,
                   name=None, ax=None, **kwargs):
    fpr, tpr, _ = roc_curve(y, y_pred,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)
    viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    return viz.plot(ax=ax, name=name, **kwargs)