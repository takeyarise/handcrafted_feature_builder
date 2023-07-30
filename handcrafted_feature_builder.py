from typing import Optional
from functools import partial, wraps
import numpy as np


__all__ = ["HandcraftedFeatureBuilder"]


def register_process(func):
    @wraps(func)
    def _add_method(self, **kwargs):
        if "axis" not in kwargs.keys():
            kwargs["axis"] = self.target_axis
        self.processes.append(partial(func, **kwargs))
        return self

    name = func.__name__.lower()
    setattr(HandcraftedFeatureBuilder, name, _add_method)
    return func


class HandcraftedFeatureBuilder:
    def __init__(self, target_axis: Optional[int] = None):
        self.target_axis = target_axis
        self.processes = list()

    @classmethod
    def from_str(
        cls, process_str: str, target_axis: Optional[int] = None, delimiter: str = ","
    ):
        ret = cls(target_axis=target_axis)
        processes = process_str.split(delimiter)
        for process in processes:
            getattr(ret, process.lower())()
        return ret

    def build(self):
        return self._Apply(self.processes, self.target_axis)

    class _Apply:
        def __init__(self, processes, target_axis=None):
            self.processes = processes
            self.target_axis = target_axis

        def __call__(self, x):
            ret = list()
            for process in self.processes:
                ret.append(process(x))
            return np.concatenate(ret, axis=self.target_axis)


@register_process
def max(x, axis):
    return np.max(x, axis=axis, keepdims=True)


@register_process
def min(x, axis):
    return np.min(x, axis=axis, keepdims=True)


@register_process
def mean(x, axis):
    return np.mean(x, axis=axis, keepdims=True)


@register_process
def std(x, axis):
    return np.std(x, axis=axis, keepdims=True)


@register_process
def var(x, axis):
    return np.var(x, axis=axis, keepdims=True)


@register_process
def sum(x, axis):
    return np.sum(x, axis=axis, keepdims=True)


@register_process
def median(x, axis):
    return np.median(x, axis=axis, keepdims=True)


@register_process
def percentile(x, axis, q):
    """percentile

    This function is not supported `HandcraftedFeatureBuilder.from_str`.

    Parameters
    ----------
    x: np.ndarray
    axis: int
    q: float
        0 <= q <= 100
    """
    return np.percentile(x, q=q, axis=axis, keepdims=True)


@register_process
def cv(x, axis):
    """coefficient of variation

    Parameters
    ----------
    x: np.ndarray
    axis: Optional[int]

    Returns
    -------
    float:
        coefficient of variation
    """
    return np.sqrt(np.var(x, axis=axis, keepdims=True)) / np.mean(
        x, axis=axis, keepdims=True
    )


@register_process
def abs_mean(x, axis):
    """means of absolute values

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    return np.mean(np.abs(x), axis=axis, keepdims=True)


@register_process
def abs_max(x, axis):
    """max of absolute values

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    return np.max(np.abs(x), axis=axis, keepdims=True)


@register_process
def abs_min(x, axis):
    """min of absolute values

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    return np.min(np.abs(x), axis=axis, keepdims=True)


@register_process
def abs_std(x, axis):
    """std of absolute values

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    return np.std(np.abs(x), axis=axis, keepdims=True)


@register_process
def intensity(x, axis):
    """intensity

    $$
    \frac{1}{n-1}\sum_{i=1}^{n-1}|x_i - x_{i+1}|
    $$

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    src = np.abs(np.diff(x, axis=axis))  # n-1 される
    return np.mean(src, axis=axis, keepdims=True)


@register_process
def zcr(x, axis):
    """zero-crossing rate

    Parameters
    ----------
    x: np.ndarray
    axis: int
    """
    src = x - np.mean(x, axis=axis, keepdims=True)
    num = np.count_nonzero(np.diff(np.sign(src), axis=axis), axis=axis, keepdims=True)
    return num / x.shape[axis]


if __name__ == "__main__":
    x = np.random.rand(2, 3, 10)
    builder = HandcraftedFeatureBuilder(target_axis=-1)
    builder = builder.max()
    builder.min().percentile(q=50).median()
    trans = builder.build()
    t = trans(x)
    # assert t.shape == (2, 3, 2)
    print(f"{t.shape=}")
    print(f"{t}")

    x = np.random.rand(2, 10)
    builder = HandcraftedFeatureBuilder.from_str("max,min,abs_min", target_axis=-1)
    trans = builder.build()
    t = trans(x)
    assert t.shape == (2, 3)
    print(f"{t.shape=}")
    print(f"{t}")

    print("===")
    x = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    # res = cv(x, -1)
    # res = intensity(x, -1)
    res = zcr(x, -1)
    print(f"{res.shape=}")
    print(res)
