"""Data schema for the nerve conduction study (NCS) records.

Every patient is one flat record (a dict / a pandas row). NCS columns are named
``{side}_{study}_{param}``, e.g. ``R_medM_lat`` = right median motor distal latency.
"""
from __future__ import annotations

from dataclasses import dataclass

SIDES = {"R": "Right", "L": "Left"}


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    unit: str
    direction: str  # "max": abnormal if value > limit; "min": abnormal if value < limit


@dataclass(frozen=True)
class Study:
    key: str
    name: str
    recording: str
    kind: str  # "motor" | "sensory"
    has_f_wave: bool = False

    @property
    def params(self) -> list[Param]:
        if self.kind == "motor":
            base = list(MOTOR_PARAMS)
            if self.has_f_wave:
                base.append(F_WAVE)
            return base
        return list(SENSORY_PARAMS)

    @property
    def nerve(self) -> str:
        return self.name.split()[0]


MOTOR_PARAMS = (
    Param("lat", "Distal latency", "ms", "max"),
    Param("amp", "CMAP amplitude", "mV", "min"),
    Param("cv", "Conduction velocity", "m/s", "min"),
)
F_WAVE = Param("f", "F-wave min. latency", "ms", "max")
SENSORY_PARAMS = (
    Param("lat", "Peak latency", "ms", "max"),
    Param("amp", "SNAP amplitude", "uV", "min"),
    Param("cv", "Conduction velocity", "m/s", "min"),
)

STUDIES: tuple[Study, ...] = (
    Study("medM", "Median motor", "APB; stim wrist/elbow", "motor", True),
    Study("ulnM", "Ulnar motor", "ADM; stim wrist/below elbow", "motor", True),
    Study("radM", "Radial motor", "EIP; stim forearm/elbow", "motor", False),
    Study("medS", "Median sensory", "Digit II, antidromic 14 cm", "sensory"),
    Study("ulnS", "Ulnar sensory", "Digit V, antidromic 14 cm", "sensory"),
    Study("radS", "Radial sensory", "Snuffbox, superficial radial 10 cm", "sensory"),
)
STUDY_BY_KEY = {s.key: s for s in STUDIES}

# Commonly cited adult textbook limits. Replace with laboratory / population-specific
# normative data before using for research conclusions.
DEFAULT_NORMALS: dict[str, float] = {
    "medM_lat": 4.4, "medM_amp": 4.0, "medM_cv": 49, "medM_f": 31,
    "ulnM_lat": 3.3, "ulnM_amp": 6.0, "ulnM_cv": 49, "ulnM_f": 32,
    "radM_lat": 2.9, "radM_amp": 2.0, "radM_cv": 49,
    "medS_lat": 3.5, "medS_amp": 20, "medS_cv": 50,
    "ulnS_lat": 3.1, "ulnS_amp": 17, "ulnS_cv": 50,
    "radS_lat": 2.9, "radS_amp": 15, "radS_cv": 50,
}

NDI_SECTIONS = (
    "Pain intensity", "Personal care", "Lifting", "Reading", "Headaches",
    "Concentration", "Work", "Driving", "Sleeping", "Recreation",
)

DEMOGRAPHIC_COLS = [
    "study_id", "age", "sex", "date", "height_cm", "weight_kg", "occupation",
    "pain_side", "pain_duration_months", "vas", "paraesthesia", "limb_temp_c",
]
NDI_COLS = [f"ndi_q{i + 1}" for i in range(10)]
IPAQ_COLS = ["ipaq_vig_days", "ipaq_vig_min", "ipaq_mod_days", "ipaq_mod_min",
             "ipaq_walk_days", "ipaq_walk_min", "ipaq_sit_min"]
BC_COLS = ["bc_fat_pct", "bc_muscle_kg", "bc_visceral", "bc_ffm_kg", "bc_tbw_l", "bc_bmr_kcal"]


def ncs_columns() -> list[str]:
    return [f"{sd}_{s.key}_{p.key}" for sd in SIDES for s in STUDIES for p in s.params]


ALL_INPUT_COLS = DEMOGRAPHIC_COLS + NDI_COLS + IPAQ_COLS + BC_COLS + ncs_columns()
