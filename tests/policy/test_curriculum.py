"""
Tests for M7 CurriculumManager.

No GPU required — pure arithmetic on step counts.
Tests confirm phase triggers at the correct step counts per the MSD.
"""

import pytest
from atar.policy.curriculum import CurriculumManager
from atar.policy.types import CurriculumPhase


def test_phase_1_at_start() -> None:
    """Phase 1 must be active from step 0."""
    cm = CurriculumManager(total_steps=1000)
    assert cm.get_phase(0) == CurriculumPhase.PHASE_1


def test_phase_boundaries_at_exact_fractions() -> None:
    """
    Phase 2 starts at 30% of total_steps, Phase 3 at 70%.
    Verify boundary steps exactly (using total_steps=1000 for clean fractions).
    """
    cm = CurriculumManager(total_steps=1000)
    p2, p3 = cm.phase_boundaries()
    assert p2 == 300, f"Phase 2 should start at step 300, got {p2}"
    assert p3 == 700, f"Phase 3 should start at step 700, got {p3}"


def test_phase_2_start_step() -> None:
    """Step just before Phase 2 boundary is Phase 1; at boundary is Phase 2."""
    cm = CurriculumManager(total_steps=1000)
    assert cm.get_phase(299) == CurriculumPhase.PHASE_1
    assert cm.get_phase(300) == CurriculumPhase.PHASE_2


def test_phase_3_start_step() -> None:
    """Step just before Phase 3 boundary is Phase 2; at boundary is Phase 3."""
    cm = CurriculumManager(total_steps=1000)
    assert cm.get_phase(699) == CurriculumPhase.PHASE_2
    assert cm.get_phase(700) == CurriculumPhase.PHASE_3


def test_phase_3_at_end() -> None:
    """Last step must be Phase 3."""
    cm = CurriculumManager(total_steps=1000)
    assert cm.get_phase(999) == CurriculumPhase.PHASE_3
    assert cm.get_phase(1000) == CurriculumPhase.PHASE_3


def test_allowed_tiers_per_phase() -> None:
    """MSD-specified allowed tiers per phase."""
    cm = CurriculumManager(total_steps=1000)
    assert cm.allowed_tiers(CurriculumPhase.PHASE_1) == [1, 2]
    assert cm.allowed_tiers(CurriculumPhase.PHASE_2) == [1, 2, 3, 4]
    assert cm.allowed_tiers(CurriculumPhase.PHASE_3) == [1, 2, 3, 4, 5]


def test_should_transition_detects_crossings() -> None:
    """should_transition() is True only on the exact crossing steps."""
    cm = CurriculumManager(total_steps=1000)

    # No transition at step 0
    assert not cm.should_transition(0)

    # Not at step 299 (still phase 1)
    assert not cm.should_transition(299)

    # Transition AT step 300 (phase 1 → phase 2)
    assert cm.should_transition(300)

    # Not at step 301
    assert not cm.should_transition(301)

    # Not at step 699
    assert not cm.should_transition(699)

    # Transition AT step 700 (phase 2 → phase 3)
    assert cm.should_transition(700)

    # Not at step 701
    assert not cm.should_transition(701)


def test_invalid_total_steps() -> None:
    """total_steps <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        CurriculumManager(total_steps=0)
    with pytest.raises(ValueError):
        CurriculumManager(total_steps=-1)


def test_curriculum_with_odd_total_steps() -> None:
    """
    Mocked short training run: confirm phase switches at correct points
    for total_steps=50 (used in tests that simulate a short training session).
    """
    cm = CurriculumManager(total_steps=50)
    p2, p3 = cm.phase_boundaries()
    assert p2 == 15   # int(50 * 0.3) = 15
    assert p3 == 35   # int(50 * 0.7) = 35

    assert cm.get_phase(14) == CurriculumPhase.PHASE_1
    assert cm.get_phase(15) == CurriculumPhase.PHASE_2
    assert cm.get_phase(34) == CurriculumPhase.PHASE_2
    assert cm.get_phase(35) == CurriculumPhase.PHASE_3
