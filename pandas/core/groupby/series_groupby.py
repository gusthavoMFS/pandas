"""
Define the SeriesGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a SeriesGroupBy object.
"""


from __future__ import annotations

from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    Interval,
    lib,
)
from pandas.errors import SpecificationError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_int64,
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)

from pandas.core.dtypes.missing import (
    notna,
)

from pandas.core import algorithms
from pandas.core.apply import (
    maybe_mangle_lambdas,
    validate_func_kwargs,
    warn_alias_replacement,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
    base,
    ops,
)
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
    _agg_template_series,
    _apply_docs,
    _transform_template,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.series import Series
from pandas.core.util.numba_ import maybe_use_numba

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        FillnaOptions,
        Manager,
        SingleManager,
        TakeIndexer,
    )

    from pandas import Categorical


class SeriesGroupBy(GroupBy[Series]):
    def _wrap_agged_manager(self, mgr: Manager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> SingleManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and not is_numeric_dtype(ser.dtype):
            # GH#41291 match Series behavior
            kwd_name = "numeric_only"
            raise TypeError(
                f"Cannot use {kwd_name}=True with "
                f"{type(self).__name__}.{name} and non-numeric dtypes."
            )
        return single

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])

    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).min()
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg('min')
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
       min  max
    1    1    2
    2    3    4

    The output column names can be controlled by passing
    the desired column names and aggregations as keyword arguments.

    >>> s.groupby([1, 1, 2, 2]).agg(
    ...     minimum='min',
    ...     maximum='max',
    ... )
       minimum  maximum
    1        1        2
    2        3        4

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
    1    1.0
    2    3.0
    dtype: float64
    """
    )

    @Appender(
        _apply_docs["template"].format(
            input="series", examples=_apply_docs["series_examples"]
        )
    )
    def apply(self, func, *args, **kwargs) -> Series:
        return super().apply(func, *args, **kwargs)

    @doc(_agg_template_series, examples=_agg_examples_doc, klass="Series")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                # Not all agg functions support numba, only propagate numba kwargs
                # if user asks for numba, and engine is not None
                # (if engine is None, the called function will handle the case where
                # numba is requested via the global option)
                kwargs["engine"] = engine
            if engine_kwargs is not None:
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)

        elif isinstance(func, abc.Iterable):
            # Catch instances of lists / tuples
            # but not the class list / tuple itself.
            func = maybe_mangle_lambdas(func)
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                # columns is not narrowed by mypy from relabeling flag
                assert columns is not None  # for mypy
                ret.columns = columns
            if not self.as_index:
                ret = ret.reset_index()
            return ret

        else:
            cyfunc = com.get_cython_func(func)
            if cyfunc and not args and not kwargs:
                warn_alias_replacement(self, func, cyfunc)
                return getattr(self, cyfunc)()

            if maybe_use_numba(engine):
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )

            if self.ngroups == 0:
                # e.g. test_evaluate_with_empty_groups without any groups to
                #  iterate over, we have no output on which to do dtype
                #  inference. We default to using the existing dtype.
                #  xref GH#51445
                obj = self._obj_with_exclusions
                return self.obj._constructor(
                    [],
                    name=self.obj.name,
                    index=self.grouper.agg_index,
                    dtype=obj.dtype,
                )

            if self.grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)

            try:
                return self._python_agg_general(func, *args, **kwargs)
            except KeyError:
                # KeyError raised in test_groupby.test_basic is bc the func does
                #  a dictionary lookup on group.name, but group name is not
                #  pinned in _python_agg_general, only in _aggregate_named
                result = self._aggregate_named(func, *args, **kwargs)

                warnings.warn(
                    "Pinning the groupby key to each group in "
                    f"{type(self).__name__}.agg is deprecated, and cases that "
                    "relied on it will raise in a future version. "
                    "If your operation requires utilizing the groupby keys, "
                    "iterate over the groupby object instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

                # result is a dict whose keys are the elements of agg_index
                result = Series(result, index=self.grouper.agg_index)
                result = self._wrap_aggregated_output(result)
                return result

    agg = aggregate

    def _python_agg_general(self, func, *args, **kwargs):
        orig_func = func
        func = com.is_builtin_func(func)
        if orig_func != func:
            alias = com._builtin_table_alias[func]
            warn_alias_replacement(self, orig_func, alias)
        f = lambda x: func(x, *args, **kwargs)

        obj = self._obj_with_exclusions
        result = self.grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg, *args, **kwargs) -> DataFrame:
        if isinstance(arg, dict):
            if self.as_index:
                # GH 15931
                raise SpecificationError("nested renamer is not supported")
            else:
                # GH#50684 - This accidentally worked in 1.x
                msg = (
                    "Passing a dictionary to SeriesGroupBy.agg is deprecated "
                    "and will raise in a future version of pandas. Pass a list "
                    "of aggregations instead."
                )
                warnings.warn(
                    message=msg,
                    category=FutureWarning,
                    stacklevel=find_stack_level(),
                )
                arg = list(arg.items())
        elif any(isinstance(x, (tuple, list)) for x in arg):
            arg = [(x, x) if not isinstance(x, (tuple, list)) else x for x in arg]
        else:
            # list of functions / function names
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)

        results: dict[base.OutputKey, DataFrame | Series] = {}
        with com.temp_setattr(self, "as_index", True):
            # Combine results using the index, need to adjust index after
            # if as_index=False (GH#50724)
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)

        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results]
            )
            return res_df

        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)

        return output

    def _wrap_applied_output(
        self,
        data: Series,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> DataFrame | Series:
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        values : List[Any]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(values) == 0:
            # GH #6265
            if is_transform:
                # GH#47787 see test_group_on_empty_multiindex
                res_index = data.index
            else:
                res_index = self.grouper.agg_index

            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=res_index,
                dtype=data.dtype,
            )
        assert values is not None

        if isinstance(values[0], dict):
            # GH #823 #24880
            index = self.grouper.agg_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_df = self._reindex_output(res_df)
            # if self.observed is False,
            # keep all-NaN rows created while re-indexing
            res_ser = res_df.stack(future_stack=True)
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            # GH #6265 #24880
            result = self.obj._constructor(
                data=values, index=self.grouper.agg_index, name=self.obj.name
            )
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return self._reindex_output(result)

    def _aggregate_named(self, func, *args, **kwargs):
        # Note: this is very similar to _aggregate_series_pure_python,
        #  but that does not pin group.name
        result = {}
        initialized = False

        for name, group in self.grouper.get_iterator(
            self._obj_with_exclusions, axis=self.axis
        ):
            # needed for pandas/tests/groupby/test_groupby.py::test_basic_aggregations
            object.__setattr__(group, "name", name)

            output = func(group, *args, **kwargs)
            output = ops.extract_result(output)
            if not initialized:
                # We only do this validation on the first iteration
                ops.check_result_array(output, group.dtype)
                initialized = True
            result[name] = output

        return result

    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series(
    ...    [390.0, 350.0, 30.0, 20.0],
    ...    index=["Falcon", "Falcon", "Parrot", "Parrot"],
    ...    name="Max Speed")
    >>> grouped = ser.groupby([1, 1, 2, 2])
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
        Falcon    0.707107
        Falcon   -0.707107
        Parrot    0.707107
        Parrot   -0.707107
        Name: Max Speed, dtype: float64

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
    Falcon    40.0
    Falcon    40.0
    Parrot    10.0
    Parrot    10.0
    Name: Max Speed, dtype: float64

    >>> grouped.transform("mean")
    Falcon    370.0
    Falcon    370.0
    Parrot     25.0
    Parrot     25.0
    Name: Max Speed, dtype: float64

    .. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

    >>> grouped.transform(lambda x: x.astype(int).max())
    Falcon    390
    Falcon    390
    Parrot     30
    Parrot     30
    Name: Max Speed, dtype: int64
    """
    )

    @Substitution(klass="Series", example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _cython_transform(
        self, how: str, numeric_only: bool = False, axis: AxisInt = 0, **kwargs
    ):
        assert axis == 0  # handled by caller

        obj = self._obj_with_exclusions

        try:
            result = self.grouper._cython_operation(
                "transform", obj._values, how, axis, **kwargs
            )
        except NotImplementedError as err:
            # e.g. test_groupby_raises_string
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(
        self, func: Callable, engine, engine_kwargs, *args, **kwargs
    ) -> Series:
        """
        Transform with a callable `func`.
        """
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        assert callable(func)
        klass = type(self.obj)

        results = []
        for name, group in self.grouper.get_iterator(
            self._obj_with_exclusions, axis=self.axis
        ):
            # this setattr is needed for test_transform_lambda_with_datetimetz
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # check for empty "results" to avoid concat ValueError
        if results:
            from pandas.core.reshape.concat import concat

            concatenated = concat(results)
            result = self._set_agg_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)

        result.name = self.obj.name
        return result

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.

        Returns
        -------
        Series

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return notna(b) and b

        try:
            indices = [
                self._get_index(name)
                for name, group in self.grouper.get_iterator(
                    self._obj_with_exclusions, axis=self.axis
                )
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series | DataFrame:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        Examples
        --------
        For SeriesGroupby:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    3
        dtype: int64
        >>> ser.groupby(level=0).nunique()
        a    2
        b    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 3], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample('MS').nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        ids, _, _ = self.grouper.group_info

        val = self.obj._values

        codes, _ = algorithms.factorize(val, sort=False)
        sorter = np.lexsort((codes, ids))
        codes = codes[sorter]
        ids = ids[sorter]

        # group boundaries are where group ids change
        # unique observations are where sorted values change
        idx = np.r_[0, 1 + np.nonzero(ids[1:] != ids[:-1])[0]]
        inc = np.r_[1, codes[1:] != codes[:-1]]

        # 1st item of each group is a new unique observation
        mask = codes == -1
        if dropna:
            inc[idx] = 1
            inc[mask] = 0
        else:
            inc[mask & np.r_[False, mask[:-1]]] = 0
            inc[idx] = 1

        out = np.add.reduceat(inc, idx).astype("int64", copy=False)
        if len(ids):
            # NaN/NaT group exists if the head of ids is -1,
            # so remove it from res and exclude its index from idx
            if ids[0] == -1:
                res = out[1:]
                idx = idx[np.flatnonzero(idx)]
            else:
                res = out
        else:
            res = out[1:]
        ri = self.grouper.agg_index

        # we might have duplications among the bins
        if len(res) != len(ri):
            res, out = np.zeros(len(ri), dtype=out.dtype), res
            if len(ids) > 0:
                # GH#21334s
                res[ids[idx]] = out

        result: Series | DataFrame = self.obj._constructor(
            res, index=ri, name=self.obj.name
        )
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return self._reindex_output(result, fill_value=0)

    @doc(Series.describe)
    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        return super().describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series | DataFrame:
        name = "proportion" if normalize else "count"

        if bins is None:
            result = self._value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )
            result.name = name
            return result

        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut

        ids, _, _ = self.grouper.group_info
        val = self.obj._values

        index_names = self.grouper.names + [self.obj.name]

        if isinstance(val.dtype, CategoricalDtype) or (
            bins is not None and not np.iterable(bins)
        ):
            # scalar bins cannot be done at top level
            # in a backward compatible way
            # GH38672 relates to categorical dtype
            ser = self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
            )
            ser.name = name
            ser.index.names = index_names
            return ser

        # groupby removes null keys from groupings
        mask = ids != -1
        ids, val = ids[mask], val[mask]

        lab: Index | np.ndarray
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            # lab is a Categorical with categories an IntervalIndex
            cat_ser = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast("Categorical", cat_ser._values)
            lev = cat_obj.categories
            lab = lev.take(
                cat_obj.codes,
                allow_fill=True,
                fill_value=lev._na_value,
            )
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]

        if isinstance(lab.dtype, IntervalDtype):
            # TODO: should we do this inside II?
            lab_interval = cast(Interval, lab)

            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))

        ids, lab = ids[sorter], lab[sorter]

        # group boundaries are where group ids change
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges

        # new values are where sorted labels change
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True  # group boundaries are also new values
        out = np.diff(np.nonzero(np.r_[inc, True])[0])  # value counts

        # num. of times each group should be repeated
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))

        # multi-index components
        codes = self.grouper.reconstructed_codes
        codes = [rep(level_codes) for level_codes in codes] + [llab(lab, inc)]
        levels = [ping.group_index for ping in self.grouper.groupings] + [lev]

        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]

        if normalize:
            out = out.astype("float")
            d = np.diff(np.r_[idx, len(ids)])
            if dropna:
                m = ids[lab == -1]
                np.add.at(d, m, -1)
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc

        if sort and bins is None:
            cat = ids[inc][mask] if dropna else ids[inc]
            sorter = np.lexsort((out if ascending else -out, cat))
            out, codes[-1] = out[sorter], codes[-1][sorter]

        if bins is not None:
            # for compat. with libgroupby.value_counts need to ensure every
            # bin is present at every index level, null filled with zeros
            diff = np.zeros(len(out), dtype="bool")
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]

            ncat, nbin = diff.sum(), len(levels[-1])

            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]

            right = [diff.cumsum() - 1, codes[-1]]

            # error: Argument 1 to "get_join_indexers" has incompatible type
            # "List[ndarray[Any, Any]]"; expected "List[Union[Union[ExtensionArray,
            # ndarray[Any, Any]], Index, Series]]
            _, idx = get_join_indexers(
                left, right, sort=False, how="left"  # type: ignore[arg-type]
            )
            out = np.where(idx != -1, out[idx], 0)

            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]

            # build the multi-index w/ full levels
            def build_codes(lev_codes: np.ndarray) -> np.ndarray:
                return np.repeat(lev_codes[diff], nbin)

            codes = [build_codes(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])

        mi = MultiIndex(
            levels=levels, codes=codes, names=index_names, verify_integrity=False
        )

        if is_integer_dtype(out.dtype):
            out = ensure_int64(out)
        result = self.obj._constructor(out, index=mi, name=name)
        if not self.as_index:
            result = result.reset_index()
        return result

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        inplace: bool = False,
        limit: int | None = None,
        downcast: dict | None | lib.NoDefault = lib.no_default,
    ) -> Series | None:
        """
        Fill NA/NaN values using the specified method within groups.

        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list. Users wanting to use the ``value`` argument and not ``method``
            should prefer :meth:`.Series.fillna` as this
            will produce the same result and be more performant.
        method : {{'bfill', 'ffill', None}}, default None
            Method to use for filling holes. ``'ffill'`` will propagate
            the last valid observation forward within a group.
            ``'bfill'`` will use next valid observation to fill the gap.

            .. deprecated:: 2.1.0
                Use obj.ffill or obj.bfill instead.

        axis : {0 or 'index', 1 or 'columns'}
            Unused, only for compatibility with :meth:`DataFrameGroupBy.fillna`.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        inplace : bool, default False
            Broken. Do not set to True.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill within a group. In other words,
            if there is a gap with more than this number of consecutive NaNs,
            it will only be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

            .. deprecated:: 2.1.0

        Returns
        -------
        Series
            Object with missing values filled within groups.

        See Also
        --------
        ffill : Forward fill values within a group.
        bfill : Backward fill values within a group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['cat', 'cat', 'cat', 'mouse', 'mouse']
        >>> ser = pd.Series([1, None, None, 2, None], index=lst)
        >>> ser
        cat    1.0
        cat    NaN
        cat    NaN
        mouse  2.0
        mouse  NaN
        dtype: float64
        >>> ser.groupby(level=0).fillna(0, limit=1)
        cat    1.0
        cat    0.0
        cat    NaN
        mouse  2.0
        mouse  0.0
        dtype: float64
        """
        result = self._op_via_apply(
            "fillna",
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        return result

    def take(
        self,
        indices: TakeIndexer,
        axis: Axis | lib.NoDefault = lib.no_default,
        **kwargs,
    ) -> Series:
        """
        Return the elements in the given *positional* indices in each group.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        If a requested index does not exist for some group, this method will raise.
        To get similar behavior that ignores indices that don't exist, see
        :meth:`.SeriesGroupBy.nth`.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take in each group.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
            For `SeriesGroupBy` this parameter is unused and defaults to 0.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

        Returns
        -------
        Series
            A Series containing the elements taken from each group.

        See Also
        --------
        Series.take : Take elements from a Series along an axis.
        Series.loc : Select a subset of a DataFrame by labels.
        Series.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.
        SeriesGroupBy.nth : Similar to take, won't raise if indices don't exist.

        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan),
        ...                    ('rabbit', 'mammal', 15.0)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[4, 3, 2, 1, 0])
        >>> df
             name   class  max_speed
        4  falcon    bird      389.0
        3  parrot    bird       24.0
        2    lion  mammal       80.5
        1  monkey  mammal        NaN
        0  rabbit  mammal       15.0
        >>> gb = df["name"].groupby([1, 1, 2, 2, 2])

        Take elements at positions 0 and 1 along the axis 0 in each group (default).

        >>> gb.take([0, 1])
        1  4    falcon
           3    parrot
        2  2      lion
           1    monkey
        Name: name, dtype: object

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> gb.take([-1, -2])
        1  3    parrot
           4    falcon
        2  0    rabbit
           1    monkey
        Name: name, dtype: object
        """
        result = self._op_via_apply("take", indices=indices, axis=axis, **kwargs)
        return result

    def skew(
        self,
        axis: Axis | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ) -> Series:
        """
        Return unbiased skew within groups.

        Normalized by N-1.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis for the function to be applied on.
            This parameter is only for compatibility with DataFrame and is unused.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series

        See Also
        --------
        Series.skew : Return unbiased skew over requested axis.

        Examples
        --------
        >>> ser = pd.Series([390., 350., 357., np.nan, 22., 20., 30.],
        ...                 index=['Falcon', 'Falcon', 'Falcon', 'Falcon',
        ...                        'Parrot', 'Parrot', 'Parrot'],
        ...                 name="Max Speed")
        >>> ser
        Falcon    390.0
        Falcon    350.0
        Falcon    357.0
        Falcon      NaN
        Parrot     22.0
        Parrot     20.0
        Parrot     30.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew()
        Falcon    1.525174
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew(skipna=False)
        Falcon         NaN
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        """
        if axis is lib.no_default:
            axis = 0

        if axis != 0:
            result = self._op_via_apply(
                "skew",
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
            return result

        def alt(obj):
            # This should not be reached since the cython path should raise
            #  TypeError and not NotImplementedError.
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            "skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(Series.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(Series.nlargest.__doc__)
    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest.__doc__)
    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.idxmin.__doc__)
    def idxmin(
        self, axis: Axis | lib.NoDefault = lib.no_default, skipna: bool = True
    ) -> Series:
        return self._idxmax_idxmin("idxmin", axis=axis, skipna=skipna)

    @doc(Series.idxmax.__doc__)
    def idxmax(
        self, axis: Axis | lib.NoDefault = lib.no_default, skipna: bool = True
    ) -> Series:
        return self._idxmax_idxmin("idxmax", axis=axis, skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> Series:
        result = self._op_via_apply(
            "corr", other=other, method=method, min_periods=min_periods
        )
        return result

    @doc(Series.cov.__doc__)
    def cov(
        self, other: Series, min_periods: int | None = None, ddof: int | None = 1
    ) -> Series:
        result = self._op_via_apply(
            "cov", other=other, min_periods=min_periods, ddof=ddof
        )
        return result

    @property
    def is_monotonic_increasing(self) -> Series:
        """
        Return whether each group's values are monotonically increasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'])
        >>> s.groupby(level=0).is_monotonic_increasing
        Falcon    False
        Parrot     True
        dtype: bool
        """
        return self.apply(lambda ser: ser.is_monotonic_increasing)

    @property
    def is_monotonic_decreasing(self) -> Series:
        """
        Return whether each group's values are monotonically decreasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'])
        >>> s.groupby(level=0).is_monotonic_decreasing
        Falcon     True
        Parrot    False
        dtype: bool
        """
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    @doc(Series.hist.__doc__)
    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        result = self._op_via_apply(
            "hist",
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    @property
    @doc(Series.dtype.__doc__)
    def dtype(self) -> Series:
        return self.apply(lambda ser: ser.dtype)

    def unique(self) -> Series:
        """
        Return unique values for each group.

        It returns unique values for each of the grouped values. Returned in
        order of appearance. Hash table-based unique, therefore does NOT sort.

        Returns
        -------
        Series
            Unique values for each of the grouped values.

        See Also
        --------
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> df = pd.DataFrame([('Chihuahua', 'dog', 6.1),
        ...                    ('Beagle', 'dog', 15.2),
        ...                    ('Chihuahua', 'dog', 6.9),
        ...                    ('Persian', 'cat', 9.2),
        ...                    ('Chihuahua', 'dog', 7),
        ...                    ('Persian', 'cat', 8.8)],
        ...                   columns=['breed', 'animal', 'height_in'])
        >>> df
               breed     animal   height_in
        0  Chihuahua        dog         6.1
        1     Beagle        dog        15.2
        2  Chihuahua        dog         6.9
        3    Persian        cat         9.2
        4  Chihuahua        dog         7.0
        5    Persian        cat         8.8
        >>> ser = df.groupby('animal')['breed'].unique()
        >>> ser
        animal
        cat              [Persian]
        dog    [Chihuahua, Beagle]
        Name: breed, dtype: object
        """
        result = self._op_via_apply("unique")
        return result
