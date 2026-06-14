"""Death certificate pipeline tools."""

from .death_certificate_consistency import (
    analyze_death_certificate_consistency,
    analyze_death_certificate_consistency_base64,
)

__all__ = [
    "analyze_death_certificate_consistency",
    "analyze_death_certificate_consistency_base64",
]
