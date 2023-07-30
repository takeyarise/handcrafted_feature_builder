from typing import Optional
from functools import partial
import numpy as np


__all__ = ['HandcraftedFeatureBuilder']


def register_process(func):
    def _add_method(self, **kwargs):
        self.processes.append(
            partial(func, axis=self.target_axis, **kwargs)
        )
        return self

    name = func.__name__.capitalize()
    setattr(HandcraftedFeatureBuilder, name, _add_method)
    return func


class HandcraftedFeatureBuilder:
    def __init__(self, target_axis: Optional[int]=None):
        self.target_axis = target_axis
        self.processes = list()

    @classmethod
    def from_str(cls, process_str: str, target_axis: Optional[int]=None, delimiter: str=','):
        ret = cls(target_axis=target_axis)
        process_str = process_str.split(delimiter)
        for process in process_str:
            getattr(ret, process.capitalize())()
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


# NOTE: 以下を追加しても `from_str` が対応していない
# @register_process
# def percentile(x, axis, q):
#     return np.percentile(x, q=q, axis=axis, keepdims=True)


if __name__ == '__main__':
    x = np.random.rand(2, 3, 10)
    builder = HandcraftedFeatureBuilder(target_axis=-1)
    builder = builder.Max()
    builder.Min()
    trans = builder.build()
    t = trans(x)
    assert t.shape == (2, 3, 2)
    print(f'{t}')

    x = np.random.rand(2, 10)
    builder = HandcraftedFeatureBuilder.from_str('max,min', target_axis=-1)
    trans = builder.build()
    t = trans(x)
    assert t.shape == (2, 2)
    print(f'{t}')
