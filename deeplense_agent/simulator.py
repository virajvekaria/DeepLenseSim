from __future__ import annotations

import math
import random
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from deeplense.lens import DeepLens

from .models import (
    CapabilitySummary,
    GeneratedImageArtifact,
    ModelCapability,
    ModelConfiguration,
    ResolvedSimulationRequest,
    SimulationPlan,
    SimulationRequest,
    SimulationRunResult,
    SubstructureType,
)


@dataclass(frozen=True)
class ModelPreset:
    summary: str
    default_resolution: int
    instrument: str
    source_light_mode: str


MODEL_PRESETS: dict[ModelConfiguration, ModelPreset] = {
    ModelConfiguration.MODEL_I: ModelPreset(
        summary="Original single-band Sersic source with Gaussian PSF and noisy 150x150 imaging.",
        default_resolution=150,
        instrument="gaussian_psf",
        source_light_mode="source_light",
    ),
    ModelConfiguration.MODEL_II: ModelPreset(
        summary="Euclid-like observation model with a magnitude-based Sersic source and 64x64 imaging.",
        default_resolution=64,
        instrument="Euclid",
        source_light_mode="source_light_mag",
    ),
    ModelConfiguration.MODEL_III: ModelPreset(
        summary="HST-like observation model with a magnitude-based Sersic source and 64x64 imaging.",
        default_resolution=64,
        instrument="HST",
        source_light_mode="source_light_mag",
    ),
}


@contextmanager
def temporary_seed(seed: int | None):
    if seed is None:
        yield
        return

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    return cleaned.strip("-") or "simulation"


class DeepLenseSimulationService:
    def __init__(self, output_root: Path | str = Path("artifacts") / "deeplense_agent") -> None:
        self.output_root = Path(output_root)

    def get_capabilities(self) -> CapabilitySummary:
        supported_models = [
            ModelCapability(
                configuration=configuration,
                summary=preset.summary,
                default_resolution=preset.default_resolution,
                instrument=preset.instrument,
                source_profile=(
                    "Sersic source (amplitude-based)"
                    if preset.source_light_mode == "source_light"
                    else "Sersic source (magnitude-based)"
                ),
            )
            for configuration, preset in MODEL_PRESETS.items()
        ]
        return CapabilitySummary(
            supported_models=supported_models,
            supported_substructure_types=[
                SubstructureType.NO_SUB,
                SubstructureType.CDM,
                SubstructureType.AXION,
            ],
            notes=[
                "Model_IV is intentionally excluded here because it depends on the external Galaxy10 DECals dataset that is not bundled in this repository.",
                "Resolution defaults follow the canonical DeepLenseSim configurations, but the agent can override them for smaller or larger runs.",
            ],
        )

    def preview(self, request: SimulationRequest) -> SimulationPlan:
        preset = MODEL_PRESETS[request.configuration]
        notes: list[str] = []
        defaulted_fields: list[str] = []

        resolution = request.resolution
        if resolution is None:
            resolution = preset.default_resolution
            defaulted_fields.append("resolution")
        elif resolution != preset.default_resolution:
            notes.append(
                f"Using custom resolution {resolution} instead of the canonical {preset.default_resolution} pixels for {request.configuration.value}."
            )

        axion_mass = request.axion_mass
        vortex_mass = request.vortex_mass
        if request.substructure_type is SubstructureType.AXION:
            if axion_mass is None:
                axion_mass = 1.0e-23
                defaulted_fields.append("axion_mass")
                notes.append("Axion substructure requested without a mass; defaulting axion_mass to 1e-23.")
            if vortex_mass is None:
                vortex_mass = 3.0e10
                defaulted_fields.append("vortex_mass")
                notes.append(
                    "Axion substructure requested without a vortex mass; defaulting vortex_mass to 3e10 solar masses."
                )
        else:
            axion_mass = None
            vortex_mass = None

        output_root = request.output_root if "output_root" in request.model_fields_set else str(self.output_root)
        resolved_request = ResolvedSimulationRequest(
            **request.model_dump(exclude={"resolution", "axion_mass", "vortex_mass", "output_root"}),
            resolution=resolution,
            axion_mass=axion_mass,
            vortex_mass=vortex_mass,
            instrument=preset.instrument,
            source_light_mode=preset.source_light_mode,
            output_root=output_root,
            defaulted_fields=defaulted_fields,
        )
        return SimulationPlan(
            summary=self._plan_summary(resolved_request),
            resolved_request=resolved_request,
            notes=notes,
            estimated_artifacts=[
                "One NumPy array (.npy) per generated image",
                "One PNG preview per generated image",
                "One JSON metadata file describing the full run",
                "One PNG contact sheet covering the generated images",
            ],
            ready_to_run=True,
        )

    def run(self, request: SimulationRequest) -> SimulationRunResult:
        plan = self.preview(request)
        resolved = plan.resolved_request

        run_id = self._build_run_id(resolved)
        output_dir = (Path(resolved.output_root) / run_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

        artifacts: list[GeneratedImageArtifact] = []
        preview_images: list[Image.Image] = []
        for index in range(resolved.image_count):
            image_seed = None if resolved.seed is None else resolved.seed + index
            image = self._simulate_image(resolved, image_seed)
            artifact, preview_image = self._write_image_artifacts(output_dir, image, index, image_seed)
            artifacts.append(artifact)
            preview_images.append(preview_image)

        contact_sheet_path = self._write_contact_sheet(output_dir, preview_images)
        metadata_path = output_dir / "run_metadata.json"
        result = SimulationRunResult(
            run_id=run_id,
            output_dir=output_dir,
            metadata_path=metadata_path,
            contact_sheet_path=contact_sheet_path,
            plan=plan,
            artifacts=artifacts,
        )
        metadata_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _plan_summary(self, request: ResolvedSimulationRequest) -> str:
        return (
            f"Generate {request.image_count} {request.configuration.value} image(s) with "
            f"{request.substructure_type.value} substructure at {request.resolution}x{request.resolution} pixels, "
            f"lens_redshift={request.lens_redshift}, source_redshift={request.source_redshift}, "
            f"main_halo_mass={request.main_halo_mass:.3g}."
        )

    def _build_run_id(self, request: ResolvedSimulationRequest) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = request.run_name or f"{request.configuration.value}_{request.substructure_type.value}"
        return f"{timestamp}_{slugify(label)}"

    def _simulate_image(self, request: ResolvedSimulationRequest, seed: int | None) -> np.ndarray:
        with temporary_seed(seed):
            lens = DeepLens(
                axion_mass=request.axion_mass,
                z_halo=request.lens_redshift,
                z_gal=request.source_redshift,
            )
            lens.make_single_halo(request.main_halo_mass)

            if request.substructure_type is SubstructureType.NO_SUB:
                lens.make_no_sub()
            elif request.substructure_type is SubstructureType.CDM:
                lens.make_old_cdm(n_sub=request.cdm_subhalo_mean)
            elif request.substructure_type is SubstructureType.AXION:
                assert request.vortex_mass is not None
                lens.make_vortex(request.vortex_mass)
            else:
                raise ValueError(f"Unsupported substructure type: {request.substructure_type}")

            if request.instrument != "gaussian_psf":
                lens.set_instrument(request.instrument)

            if request.source_light_mode == "source_light":
                lens.make_source_light()
                lens.simple_sim(num_pix=request.resolution)
            else:
                lens.make_source_light_mag()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Cosmology is provided.*",
                        category=UserWarning,
                    )
                    lens.simple_sim_2(numpix=request.resolution)

        return np.asarray(lens.image_real, dtype=np.float64)

    def _write_image_artifacts(
        self,
        output_dir: Path,
        image: np.ndarray,
        index: int,
        seed: int | None,
    ) -> tuple[GeneratedImageArtifact, Image.Image]:
        base_name = f"image_{index:03d}"
        npy_path = output_dir / f"{base_name}.npy"
        png_path = output_dir / f"{base_name}.png"

        np.save(npy_path, image)
        preview_image = Image.fromarray(self._normalize_image(image), mode="L")
        preview_image.save(png_path)

        artifact = GeneratedImageArtifact(
            index=index,
            seed=seed,
            npy_path=npy_path,
            png_path=png_path,
            shape=(int(image.shape[0]), int(image.shape[1])),
            min_value=float(np.min(image)),
            max_value=float(np.max(image)),
            mean_value=float(np.mean(image)),
            std_value=float(np.std(image)),
        )
        return artifact, preview_image

    def _write_contact_sheet(self, output_dir: Path, preview_images: list[Image.Image]) -> Path | None:
        if not preview_images:
            return None

        columns = min(4, len(preview_images))
        rows = math.ceil(len(preview_images) / columns)
        tile_width, tile_height = preview_images[0].size
        canvas = Image.new("L", (columns * tile_width, rows * tile_height))
        for index, image in enumerate(preview_images):
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            canvas.paste(image, (x, y))

        contact_sheet_path = output_dir / "contact_sheet.png"
        canvas.save(contact_sheet_path)
        return contact_sheet_path

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        finite_image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(np.min(finite_image))
        max_value = float(np.max(finite_image))
        if max_value <= min_value:
            return np.zeros_like(finite_image, dtype=np.uint8)
        scaled = (finite_image - min_value) / (max_value - min_value)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
