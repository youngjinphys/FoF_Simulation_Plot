import argparse
import math
import os
from dataclasses import dataclass
from numbers import Integral, Real


# Interactive_Plot imports this module before NumPy so native thread pools are
# bounded before any numerical library initializes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@dataclass
class GlobalConfig:
    """Catalog location and physical scales required by the interactive viewer."""

    data_dir: str = "./data/FoF_R"
    data_prefix: str = "resR"
    box_size: float = 100.0
    grid_size: int = 256
    particle_mass: float = 4.27e9

    def __post_init__(self):
        if isinstance(self.data_dir, list):
            if not self.data_dir:
                raise ValueError("data_dir list cannot be empty")
            self.data_dir = self.data_dir[0]

        for name in ("data_dir", "data_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if (
            isinstance(self.box_size, bool)
            or not isinstance(self.box_size, Real)
            or not math.isfinite(self.box_size)
            or self.box_size <= 0.0
        ):
            raise ValueError("box_size must be a finite positive number")
        if (
            isinstance(self.grid_size, bool)
            or not isinstance(self.grid_size, Integral)
            or self.grid_size <= 0
        ):
            raise ValueError("grid_size must be a positive integer")
        if (
            isinstance(self.particle_mass, bool)
            or not isinstance(self.particle_mass, Real)
            or not math.isfinite(self.particle_mass)
            or self.particle_mass <= 0.0
        ):
            raise ValueError("particle_mass must be a finite positive number")

        if "FoF_E" in self.data_dir and self.data_prefix == "resR":
            self.data_prefix = "resE"
        elif "FoF_R" in self.data_dir and self.data_prefix == "resE":
            self.data_prefix = "resR"
        self.data_dir = self.data_dir.rstrip("/")

    @property
    def coord_scale(self) -> float:
        return self.box_size / self.grid_size

    @property
    def dir_path(self) -> str:
        return os.path.join(self.data_dir, self.data_prefix)


def setup_common_argparse(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        default=["./data/FoF_R"],
        help="Catalog root directories, for example ./data/FoF_R or ./data/FoF_E",
    )
    parser.add_argument("--data-prefix", type=str, default="resR")
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--particle-mass", type=float, default=4.27e9)
    return parser


def get_global_config(
    args: argparse.Namespace,
    data_dir: str | None = None,
) -> GlobalConfig:
    if data_dir is None:
        data_dir = args.data_dir[0] if isinstance(args.data_dir, list) else args.data_dir
    return GlobalConfig(
        data_dir=data_dir,
        data_prefix=args.data_prefix,
        box_size=args.box_size,
        grid_size=args.grid_size,
        particle_mass=args.particle_mass,
    )
