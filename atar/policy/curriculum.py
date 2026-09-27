"""
M7 Curriculum Manager — three-phase curriculum as specified in the M7 MSD.

Phase boundaries (fraction of total training steps):
    Phase 1 (0–30%):  Tiers 1-2 only
    Phase 2 (30–70%): Tiers 1-4
    Phase 3 (70–100%): All 5 tiers including error-recovery

The manager signals when a transition is due so PPOTrainer can save a
checkpoint *immediately before* switching phases (hard requirement in MSD).
"""

from __future__ import annotations

from atar.policy.types import CurriculumPhase


class CurriculumManager:
    """
    Determines the current curriculum phase from the global step count.

    Parameters
    ----------
    total_steps : int
        Total training steps in the full training run (not steps per phase).
    """

    # Phase boundaries as fractions of total_steps (MSD spec, exact).
    _PHASE_2_START: float = 0.30
    _PHASE_3_START: float = 0.70

    def __init__(self, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self._total_steps = total_steps
        self._phase_2_step = int(total_steps * self._PHASE_2_START)
        self._phase_3_step = int(total_steps * self._PHASE_3_START)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_phase(self, current_step: int) -> CurriculumPhase:
        """
        Return the curriculum phase for a given global training step.

        Parameters
        ----------
        current_step : int
            Current training step (0-indexed).

        Returns
        -------
        CurriculumPhase
        """
        if current_step < self._phase_2_step:
            return CurriculumPhase.PHASE_1
        elif current_step < self._phase_3_step:
            return CurriculumPhase.PHASE_2
        else:
            return CurriculumPhase.PHASE_3

    def allowed_tiers(self, phase: CurriculumPhase) -> list[int]:
        """
        Return the list of task tiers allowed for the given phase.

        Parameters
        ----------
        phase : CurriculumPhase

        Returns
        -------
        list[int]
            Tier numbers (1-5) the task sampler should include.
        """
        if phase == CurriculumPhase.PHASE_1:
            return [1, 2]
        elif phase == CurriculumPhase.PHASE_2:
            return [1, 2, 3, 4]
        else:  # PHASE_3
            return [1, 2, 3, 4, 5]

    def should_transition(self, current_step: int) -> bool:
        """
        Return True if stepping from current_step-1 to current_step triggers
        a phase boundary crossing — i.e. the phase just changed.

        PPOTrainer uses this to detect the *exact* step at which a checkpoint
        must be saved before continuing with the new phase.

        Parameters
        ----------
        current_step : int
            The step just completed (0-indexed).
        """
        if current_step <= 0:
            return False
        return self.get_phase(current_step) != self.get_phase(current_step - 1)

    def phase_boundaries(self) -> tuple[int, int]:
        """Return the (phase_2_start_step, phase_3_start_step) boundary steps."""
        return self._phase_2_step, self._phase_3_step
