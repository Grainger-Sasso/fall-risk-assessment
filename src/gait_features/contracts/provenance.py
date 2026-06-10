from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class ExtractionProvenance:
    backend: str
    stride_feature_catalog: str
    profile: str
    library_version: str
    extracted_at_utc: str

    @classmethod
    def create(
        cls,
        backend: str,
        stride_feature_catalog: str,
        profile: str,
        library_version: str,
        extracted_at_utc: Optional[str] = None,
    ) -> "ExtractionProvenance":
        timestamp = extracted_at_utc or datetime.now(timezone.utc).isoformat()
        return cls(
            backend=backend,
            stride_feature_catalog=stride_feature_catalog,
            profile=profile,
            library_version=library_version,
            extracted_at_utc=timestamp,
        )
