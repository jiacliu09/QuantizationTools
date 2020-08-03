from .null_quan import NullQuan
from .minmax_quan import MinMaxQuan, TrueMinMaxQuan
from .scale_shift_quan import ScaleShiftQuan
from .symmetry_max_quan import SymmetryMaxQuan, TrueSymmetryMaxQuan


def get_quan(strategy, qconfig, graph, params):
    if strategy == 'null':
        return NullQuan(graph=graph, params=params, **qconfig)
    elif strategy == 'minmax':
        return MinMaxQuan(graph=graph, params=params, **qconfig)
    elif strategy == 'minmax_t':
        return TrueMinMaxQuan(graph=graph, params=params, **qconfig)
    elif strategy == 'scale_shift':
        return ScaleShiftQuan(graph=graph, params=params, **qconfig)
    elif strategy == 'symmetry_max':
        return SymmetryMaxQuan(graph=graph, params=params, **qconfig)
    elif strategy == 'symmetry_max_t':
        return TrueSymmetryMaxQuan(graph=graph, params=params, **qconfig)
    else:
        raise NotImplementedError
