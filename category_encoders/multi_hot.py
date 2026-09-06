"""Multi-hot encoding of delimiter-separated categorical items."""

import numpy as np
import pandas as pd

import category_encoders.utils as util

__author__ = 'Obvious'


class MultiHotEncoder(util.UnsupervisedTransformerMixin, util.BaseEncoder):
    """Multi-hot encoding for cells that contain several delimiter-separated items.

    Where :class:`OneHotEncoder` treats each cell as one atomic category,
    MultiHotEncoder splits the cell on a delimiter and activates one binary
    column per item, so that ``'mathematics|physics'`` lights both the
    mathematics and the physics column. The set of items is learned at fit
    time; the number of output columns is fixed by the fit, so
    ``get_feature_names_out`` always matches the transform output.

    Items are the delimiter-split fragments of a cell with surrounding
    whitespace stripped; fragments that are empty after stripping are dropped.
    This means a cell such as ``'a | b'``, ``'a||b'`` or ``'a|'`` contributes
    the items ``a`` and ``b`` (or, for the last two, only ``a``), and a cell
    that is empty or consists solely of the delimiter contributes no items.
    Items cannot themselves contain the delimiter: a cell such as
    ``'Smith, John'`` with the default delimiter is stored as two items, so
    choose a delimiter that does not occur inside the items.

    The encoding is unsupervised and has no ``inverse_transform``: a multi-hot
    row does not uniquely determine the original cell.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string and categorical columns
        will be encoded.
    drop_invariant: bool
        boolean for whether to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_unknown: str
        how to handle items that were not seen at fit time. Options are
        'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.

        'error' will raise a `ValueError` at transform time if an unknown item appears.
        'return_nan' will encode a row that contains an unknown item as `np.nan` in
        every dummy column of the affected input column.
        'value' will ignore unknown items; the known items of the same cell still
        activate their columns, so a fully unknown cell becomes all zeros.
        'indicator' behaves like 'value' and additionally adds one dummy column
        per input column (in both training and test data) that is activated
        whenever an unknown item appears.
    handle_missing: str
        how to handle missing values (NaN). Options are 'error', 'return_nan',
        'value', 'ignore', and 'indicator'. The default is 'value'.

        'error' will raise a `ValueError` if a missing value is encountered.
        'return_nan' will encode a row that contains a missing value as `np.nan`
        in every dummy column of the affected input column.
        'value' will treat missing values as another valid item at fit time, so
        a missing cell activates the missing-item column.
        'ignore' will encode missing values as 0 in every dummy column,
        NOT adding an additional category.
        'indicator' behaves like 'ignore' and additionally adds one dummy column
        per input column that is activated whenever a value is missing.
    delimiter: str
        the string that separates multiple items within one cell.
        Must be a non-empty string and must not occur inside a single item.
    use_cat_names: bool
        if True, the seen item values will be included in the encoded column
        names (e.g. ``city_paris``); collisions are suffixed with '#'. If False,
        columns are named by order of first appearance (e.g. ``city_1``), which
        keeps the names stable under category relabeling.

    Example
    -------
    >>> import pandas as pd
    >>> from category_encoders import MultiHotEncoder
    >>> X = pd.DataFrame({'topic': ['math|physics', 'math', 'physics|art', 'art']})
    >>> MultiHotEncoder(use_cat_names=True).fit_transform(X)
       topic_math  topic_physics  topic_art
    0           1              1          0
    1           1              0          0
    2           0              1          1
    3           0              0          1
    """

    prefit_ordinal = False
    encoding_relation = util.EncodingRelation.ONE_TO_N_UNIQUE
    _VALID_HANDLE_MISSING = ('error', 'return_nan', 'value', 'ignore', 'indicator')
    _VALID_HANDLE_UNKNOWN = ('error', 'return_nan', 'value', 'indicator')

    def __init__(
        self,
        verbose: int = 0,
        cols: list[str] | None = None,
        drop_invariant: bool = False,
        return_df: bool = True,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        delimiter: str = '|',
        use_cat_names: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.delimiter = delimiter
        self.use_cat_names = use_cat_names

    def _fit(self, X, y=None, **kwargs):
        if not isinstance(self.delimiter, str) or not self.delimiter:
            raise ValueError(f'delimiter must be a non-empty string, got {self.delimiter!r}')
        self.mapping = [self._fit_column(X[col], col) for col in self.cols]

    def _fit_column(self, values: pd.Series, col: str) -> dict:
        """Learn the item slots of one column in order of first appearance."""
        # track distinct items; the nan slot (when handle_missing='value') keeps
        # its position among the items by order of first appearance
        slots: list[tuple[str, object]] = []
        seen: set[object] = set()
        nan_slot_seen = False
        for cell in values.astype(object).to_numpy():
            if pd.isna(cell):
                if self.handle_missing == 'value' and not nan_slot_seen:
                    nan_slot_seen = True
                    slots.append((True, None))
                continue
            for item in self._split_cell(cell):
                if item not in seen:
                    seen.add(item)
                    slots.append((False, item))

        counts: dict[str, int] = {}
        names: list[str] = []
        item_pos: dict[str, int] = {}
        nan_value_pos = None
        for is_nan_slot, item in slots:
            if is_nan_slot:
                nan_value_pos = len(names)
                suffix = 'nan'
            else:
                item_pos[item] = len(names)
                suffix = item
            names.append(self._column_name(col, suffix, len(names), counts))

        unknown_pos = None
        if self.handle_unknown == 'indicator':
            unknown_pos = len(names)
            names.append(self._column_name(col, '-1', len(names), counts))

        missing_pos = None
        if self.handle_missing == 'indicator':
            missing_pos = len(names)
            names.append(self._column_name(col, '-2', len(names), counts))

        return {
            'col': col,
            'columns': names,
            'item_pos': item_pos,
            'nan_value_pos': nan_value_pos,
            'unknown_pos': unknown_pos,
            'missing_pos': missing_pos,
        }

    def _column_name(self, col: str, suffix: str, position: int, counts: dict[str, int]) -> str:
        """Build one output column name, deduplicating collisions for item names."""
        if not self.use_cat_names:
            return f'{col}_{position + 1}'
        base = f'{col}_{suffix}'
        found = counts.get(base, 0)
        counts[base] = found + 1
        return base + '#' * found

    def _split_cell(self, cell: object) -> list[str]:
        """Split one cell into its non-empty, whitespace-stripped items."""
        return [
            item
            for item in (fragment.strip() for fragment in str(cell).split(self.delimiter))
            if item
        ]

    def _transform(self, X):
        spec_by_col = {switch['col']: switch for switch in self.mapping}
        absent = [col for col in spec_by_col if col not in set(X.columns)]
        if absent:
            raise KeyError(
                f'Columns to be encoded are missing from the input data: {sorted(absent)}'
            )

        blocks = []
        for position, col in enumerate(X.columns):
            switch = spec_by_col.get(col)
            if switch is None:
                blocks.append(X.iloc[:, position : position + 1])
            else:
                blocks.append(self._encode_column(X.iloc[:, position], switch))
        return pd.concat(blocks, axis=1)

    def _encode_column(self, values: pd.Series, switch: dict) -> pd.DataFrame:
        """Build the binary block for one column, positionally (index-safe)."""
        col = switch['col']
        columns = switch['columns']
        item_pos = switch['item_pos']
        nan_value_pos = switch['nan_value_pos']
        unknown_pos = switch['unknown_pos']
        missing_pos = switch['missing_pos']

        raw = values.astype(object).to_numpy()
        missing_mask = pd.isna(raw)
        produces_nan = self.handle_missing == 'return_nan' or self.handle_unknown == 'return_nan'
        out = np.zeros((len(raw), len(columns)), dtype=np.float64 if produces_nan else np.int64)

        missing_idx = np.flatnonzero(missing_mask)
        if missing_idx.size:
            if self.handle_missing == 'indicator' and missing_pos is not None:
                out[missing_idx, missing_pos] = 1
            elif self.handle_missing == 'return_nan':
                out[missing_idx, :] = np.nan
            elif self.handle_missing == 'value' and nan_value_pos is not None:
                out[missing_idx, nan_value_pos] = 1
            # 'ignore' (and 'value' without a fitted missing slot): leave all zeros

        for row_idx in np.flatnonzero(~missing_mask):
            items = self._split_cell(raw[row_idx])
            if self.handle_unknown != 'value':
                unknown_items = [item for item in items if item not in item_pos]
                if unknown_items:
                    if self.handle_unknown == 'error':
                        raise ValueError(
                            f'MultiHotEncoder: unknown item(s) {unknown_items!r} found in '
                            f"column {col!r} at transform time. Use handle_unknown='value' "
                            'or fit on data that covers all items.'
                        )
                    if self.handle_unknown == 'return_nan':
                        out[row_idx, :] = np.nan
                        continue
                    if self.handle_unknown == 'indicator' and unknown_pos is not None:
                        out[row_idx, unknown_pos] = 1
            for item in items:
                position = item_pos.get(item)
                if position is not None:
                    out[row_idx, position] = 1

        return pd.DataFrame(out, columns=columns, index=values.index)
