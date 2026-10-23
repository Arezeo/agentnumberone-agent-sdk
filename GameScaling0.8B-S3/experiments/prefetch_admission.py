"""Experimental candidate admission. No transport or model calls.

A past candidate may supply reusable computation for a newer observation only
when pixels and processing identity match. It never substitutes its timestamp
or frame ID for that selected observation's identity.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    frame_id: int
    received_at: float
    context: tuple
    pixels: str
    processing: str


def admit(candidate, selected, *, current_context, action_started_at):
    if selected.context != current_context or candidate.context != current_context:
        return False, 'context_changed'
    if selected.received_at <= action_started_at:
        return False, 'selected_frame_precedes_action'
    if candidate.received_at > selected.received_at:
        return False, 'candidate_from_future'
    if candidate.processing != selected.processing:
        return False, 'processing_changed'
    if candidate.pixels != selected.pixels:
        return False, 'pixels_changed'
    return True, 'exact_pixels_for_eligible_selected_frame'
