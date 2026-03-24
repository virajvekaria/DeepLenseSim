from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfiguration(str, Enum):
    MODEL_I = "Model_I"
    MODEL_II = "Model_II"
    MODEL_III = "Model_III"


class SubstructureType(str, Enum):
    NO_SUB = "no_sub"
    CDM = "cdm"
    AXION = "axion"


class ModelCapability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    configuration: ModelConfiguration = Field(
        ...,
        alias="model_config",
        serialization_alias="model_config",
    )
    summary: str
    default_resolution: int = Field(..., ge=8)
    instrument: str
    source_profile: str


class CapabilitySummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    supported_models: list[ModelCapability]
    supported_substructure_types: list[SubstructureType]
    notes: list[str] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    configuration: ModelConfiguration = Field(
        ...,
        alias="model_config",
        serialization_alias="model_config",
        description="Canonical DeepLenseSim configuration to use. Supported now: Model_I, Model_II, Model_III.",
    )
    substructure_type: SubstructureType = Field(
        ...,
        description="Dark-matter substructure family: no_sub, cdm, or axion.",
    )
    image_count: int = Field(
        default=1,
        ge=1,
        le=25,
        description="How many independent lensing images to generate in this run.",
    )
    lens_redshift: float = Field(
        default=0.5,
        gt=0,
        description="Redshift of the foreground lens halo.",
    )
    source_redshift: float = Field(
        default=1.0,
        gt=0,
        description="Redshift of the lensed source galaxy; must be greater than the lens redshift.",
    )
    resolution: int | None = Field(
        default=None,
        ge=8,
        le=512,
        description="Square image side length in pixels. If omitted, the canonical model resolution is used.",
    )
    main_halo_mass: float = Field(
        default=1.0e12,
        gt=0,
        description="Main lens halo mass in solar masses.",
    )
    axion_mass: float | None = Field(
        default=None,
        gt=0,
        description="Axion particle mass used for axion/vortex simulations.",
    )
    vortex_mass: float | None = Field(
        default=None,
        gt=0,
        description="Mass assigned to the axion vortex perturbation.",
    )
    cdm_subhalo_mean: int = Field(
        default=25,
        ge=0,
        le=1000,
        description="Mean number of CDM subhalos drawn in the field of view.",
    )
    output_root: str = Field(
        default="artifacts/deeplense_agent",
        description="Directory under which the run folder and generated artifacts will be written.",
    )
    run_name: str | None = Field(
        default=None,
        description="Optional human-friendly label that will be folded into the run directory name.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional base random seed. When set, image i uses seed + i for reproducibility.",
    )

    @model_validator(mode="after")
    def _validate_redshifts(self) -> "SimulationRequest":
        if self.source_redshift <= self.lens_redshift:
            raise ValueError("source_redshift must be greater than lens_redshift.")
        return self


class ResolvedSimulationRequest(SimulationRequest):
    model_config = ConfigDict(populate_by_name=True)

    resolution: int = Field(..., ge=8, le=512)
    instrument: str = Field(..., description="Concrete instrument profile used by the simulator.")
    source_light_mode: str = Field(
        ...,
        description="Internal DeepLenseSim source builder used for this request.",
    )
    defaulted_fields: list[str] = Field(default_factory=list)


class SimulationPlan(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    summary: str
    resolved_request: ResolvedSimulationRequest
    notes: list[str] = Field(default_factory=list)
    estimated_artifacts: list[str] = Field(default_factory=list)
    ready_to_run: bool = True


class GeneratedImageArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    index: int = Field(..., ge=0)
    seed: int | None = None
    npy_path: Path
    png_path: Path
    shape: tuple[int, int]
    min_value: float
    max_value: float
    mean_value: float
    std_value: float


class SimulationRunResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    output_dir: Path
    metadata_path: Path
    contact_sheet_path: Path | None = None
    plan: SimulationPlan
    artifacts: list[GeneratedImageArtifact]
