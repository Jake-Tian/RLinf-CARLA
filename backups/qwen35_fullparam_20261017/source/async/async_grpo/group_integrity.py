"""Dependency-free group-layout and version-window checks for async GRPO.

RLinf reshapes rewards by position, so the batch must contain contiguous whole
groups. Weight syncs within a trajectory are legal when the group's version span
stays within algorithm.staleness_threshold; versions need not be constant.
"""

from __future__ import annotations

from collections.abc import Sequence

#: RLinf's own wording, quoted so a reader can grep for the reshape that motivates this.
_REShAPE_RULE = (
    "calculate_scores reshapes this axis as (-1, group_size) and "
    "compute_grpo_advantages reshapes it again with view(-1, group_size), "
    "neither of which checks membership"
)

#: One trajectory's weight versions as (min, max); equal when it saw no sync mid-flight.
VersionBounds = tuple[float, float]


def group_count(bsz: int, group_size: int) -> int:
    """How many whole groups fit in ``bsz``; raises if the split is not whole."""
    assert group_size > 0, f"group_size must be positive, got {group_size}"
    assert bsz % group_size == 0, (
        f"the batch holds {bsz} trajectories, which is not a whole number of "
        f"groups of {group_size} ({bsz} % {group_size} = {bsz % group_size}). "
        f"{_REShAPE_RULE}, so a partial group at the end is silently averaged "
        "against whatever follows it"
    )
    return bsz // group_size


def check_groups_intact(
    bsz: int,
    version_bounds: Sequence[VersionBounds] | None,
    group_size: int,
    max_span: int = 0,
    where: str = "",
) -> int:
    """Assert the batch axis is whole groups, each drawn from one staleness window.

    Args:
        bsz: length of the batch axis, i.e. the trajectory count that
            ``calculate_scores`` will reshape.
        version_bounds: one ``(min, max)`` weight version per trajectory, in batch
            order. ``None`` skips the window check and only verifies the group
            boundaries.
        group_size: the GRPO group size.
        max_span: how many weight versions apart two members of one group may be.
            Pass the run's ``algorithm.staleness_threshold``: the actor admits a
            trajectory on ``versions.min() >= version - staleness_threshold``, so a
            batch assembled under that rule cannot legitimately span more. ``0``
            demands the constancy the synchronous arm gets for free.
        where: free-form context for the failure message.

    Returns:
        The number of groups the batch holds.
    """
    n_groups = group_count(bsz, group_size)
    if version_bounds is None:
        return n_groups

    bounds = list(version_bounds)
    suffix = f" ({where})" if where else ""
    assert len(bounds) == bsz, (
        f"got {len(bounds)} version bounds but the batch axis is {bsz}{suffix}"
    )
    for i in range(n_groups):
        members = bounds[i * group_size : (i + 1) * group_size]
        lo = min(m[0] for m in members)
        hi = max(m[1] for m in members)
        if hi - lo > max_span:
            raise AssertionError(
                f"GRPO group {i} spans {hi - lo:.0f} weight versions, past the "
                f"declared off-policy bound of {max_span}: its members range over "
                f"[{lo:.0f}, {hi:.0f}] as {[tuple(m) for m in members]}. A group's "
                f"advantage baseline is a mean over its members, so a group may not "
                f"average policies further apart than the config allows. "
                f"{_REShAPE_RULE}{suffix}"
            )
    return n_groups


def version_spread(
    version_bounds: Sequence[VersionBounds], group_size: int
) -> dict[str, float]:
    """Report version spread and legal mid-trajectory syncs without gating data."""
    bounds = list(version_bounds)
    if not bounds:
        return {}
    worst_group = 0.0
    for i in range(0, len(bounds) - group_size + 1, group_size):
        members = bounds[i : i + group_size]
        worst_group = max(
            worst_group,
            max(m[1] for m in members) - min(m[0] for m in members),
        )
    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    return {
        "trajectories": float(len(bounds)),
        "version_min": lo,
        "version_max": hi,
        "version_span": hi - lo,
        "group_span_max": worst_group,
        "mixed_traj_ratio": sum(1 for b in bounds if b[1] > b[0]) / len(bounds),
    }
