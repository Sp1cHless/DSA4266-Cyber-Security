"""Explicit sequence/bucket geometry; no process-global configuration changes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalGrid:
    sequence_seconds: int = 100
    bucket_seconds: int = 10

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.sequence_seconds, self.bucket_seconds)):
            raise ValueError("Sequence and bucket durations must be positive integer seconds")
        if self.sequence_seconds % self.bucket_seconds:
            raise ValueError("Sequence duration must be divisible by bucket duration")
        # Existing provenance stores circular shifts as int8 and occupancy as uint8.
        if not 1 <= self.timesteps <= 127:
            raise ValueError("The existing Scheme2 format supports 1 to 127 buckets")

    @property
    def timesteps(self):
        return self.sequence_seconds // self.bucket_seconds

    @property
    def window_ms(self):
        return self.sequence_seconds * 1000

    @property
    def bucket_ms(self):
        return self.bucket_seconds * 1000

    def settings(self):
        return dict(sequence_seconds=self.sequence_seconds,
                    bucket_seconds=self.bucket_seconds, timesteps=self.timesteps)


DEFAULT_GRID = TemporalGrid()
