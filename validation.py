"""Chronological research partitions; no model or network dependencies."""

from dataclasses import asdict, dataclass
import re


def _era_number(era):
    match = re.fullmatch(r"(?:era)?(\d+)", str(era))
    if match is None:
        raise ValueError(f"unsupported v5 era label: {era!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class EraSplit:
    early_stopping: tuple
    calibration: tuple
    evaluation: tuple
    embargo: tuple

    def as_dict(self):
        return asdict(self)


def chronological_split(eras, *, evaluation_eras=95, calibration_eras=95,
                        embargo_eras=4, min_early_stopping_eras=50):
    """Reserve disjoint blocks with gaps, failing on insufficient history.

    Four weekly eras are the configured gap for this v5 20-day experiment.
    Verify horizon and era spacing before using another dataset or target.
    This split cannot undo earlier inspection of historical evaluation data.
    """
    counts = (evaluation_eras, calibration_eras, min_early_stopping_eras)
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError("partition counts must be positive integers")
    if not isinstance(embargo_eras, int) or isinstance(embargo_eras, bool) or embargo_eras < 0:
        raise ValueError("embargo must be a non-negative integer")
    ordered = sorted(set(eras), key=_era_number)
    if len({_era_number(era) for era in ordered}) != len(ordered):
        raise ValueError("ambiguous numeric era labels")
    required = sum(counts) + 2 * embargo_eras
    if len(ordered) < required:
        raise ValueError(f"need at least {required} eras; received {len(ordered)}")
    hold_start = len(ordered) - evaluation_eras
    cal_end = hold_start - embargo_eras
    cal_start = cal_end - calibration_eras
    stop_end = cal_start - embargo_eras
    return EraSplit(tuple(ordered[:stop_end]), tuple(ordered[cal_start:cal_end]),
                    tuple(ordered[hold_start:]),
                    tuple(ordered[stop_end:cal_start] + ordered[cal_end:hold_start]))
