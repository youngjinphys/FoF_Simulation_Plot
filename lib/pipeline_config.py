import argparse
import math
import os
from dataclasses import dataclass
from numbers import Integral, Real


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def numeric_token(value: float, min_decimals: int = 0) -> str:
    """Compact, collision-resistant numeric token for output paths."""
    numeric_value = float(value)
    token = format(numeric_value, ".10g")
    if math.isfinite(numeric_value):
        mantissa, exponent = format(numeric_value, ".9e").split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        scientific = f"{mantissa}e{int(exponent)}"
        if len(scientific) < len(token):
            token = scientific
    if min_decimals > 0 and "e" not in token.lower():
        if "." not in token:
            token += "."
        decimals = len(token.split(".", 1)[1])
        token += "0" * max(0, min_decimals - decimals)
    return token


@dataclass
class GlobalConfig:
    """Configuration consumed by the interactive FoF halo visualizer."""

    data_dir: str = "./data/FoF_R"
    data_prefix: str = "resR"
    output_root: str = "./output"
    num_files: int = 500
    num_workers: int = 5

    box_size: float = 100.0
    grid_size: int = 256
    particle_mass: float = 4.27e9
    min_particles: int = 20
    shape_min_particles: int = 100

    slice_min_particles: int = 100
    slice_grid: int = 1000
    num_slices: int = 10
    l_3d: float = 3.0
    line_length_mode: str = "fixed"
    plot_dpi: int = 600

    random_seed: int = 42
    filament_method: str | None = None
    filament_mst_dist: float = -1.0
    filament_mst_mean_sep_factor: float = 0.3
    filament_tweb_th: float = 5e-12
    line_only: bool = True
    draw_shape_lines: bool = False
    show_all_mass: bool = True
    mst_multi_scan: bool = False
    ortho_view_bounds: tuple | None = None

    def __post_init__(self):
        if isinstance(self.data_dir, list):
            if not self.data_dir:
                raise ValueError("data_dir list cannot be empty")
            self.data_dir = self.data_dir[0]

        for name in ("data_dir", "data_prefix", "output_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        def require_positive_real(name):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")

        def require_positive_int(name):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        def require_nonnegative_int(name):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        for name in (
            "box_size",
            "particle_mass",
            "l_3d",
            "filament_mst_mean_sep_factor",
        ):
            require_positive_real(name)
        for name in (
            "grid_size",
            "num_files",
            "num_workers",
            "slice_grid",
            "num_slices",
            "plot_dpi",
        ):
            require_positive_int(name)
        for name in ("min_particles", "shape_min_particles", "slice_min_particles"):
            require_nonnegative_int(name)

        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, Integral):
            raise ValueError("random_seed must be an integer")
        if self.line_length_mode not in {"fixed", "uniform"}:
            raise ValueError("line_length_mode must be 'fixed' or 'uniform'")
        if self.filament_method not in {"MST", "T-web", "all", None}:
            raise ValueError("filament_method must be 'MST', 'T-web', 'all', or None")
        if (
            isinstance(self.filament_mst_dist, bool)
            or not isinstance(self.filament_mst_dist, Real)
            or not math.isfinite(self.filament_mst_dist)
            or self.filament_mst_dist == 0.0
        ):
            raise ValueError(
                "filament_mst_dist must be negative for auto mode or positive"
            )
        if (
            isinstance(self.filament_tweb_th, bool)
            or not isinstance(self.filament_tweb_th, Real)
            or not math.isfinite(self.filament_tweb_th)
        ):
            raise ValueError("filament_tweb_th must be finite")
        if self.ortho_view_bounds is not None:
            if len(self.ortho_view_bounds) != 6 or not all(
                isinstance(value, Real) and math.isfinite(value)
                for value in self.ortho_view_bounds
            ):
                raise ValueError("ortho_view_bounds must contain six finite numbers")

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

    def output_dir(self, task_name: str) -> str:
        dataset_label = os.path.basename(self.data_dir)
        path = os.path.join(self.output_root, dataset_label, task_name)
        os.makedirs(path, exist_ok=True)
        return path

    def output_filename(self, plot_name: str, ext: str = "png") -> str:
        dataset_label = os.path.basename(self.data_dir)
        return f"{plot_name}_{dataset_label}.{ext}"


def setup_common_argparse(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        default=["./data/FoF_R"],
        help="Data directories (for example ./data/FoF_R or ./data/FoF_E)",
    )
    parser.add_argument("--data-prefix", type=str, default="resR")
    parser.add_argument("--output-root", type=str, default="./output")
    parser.add_argument("--nfiles", type=int, default=500)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--particle-mass", type=float, default=4.27e9)
    parser.add_argument("--min-particles", type=int, default=20)
    parser.add_argument("--shape-min-particles", type=int, default=100)
    parser.add_argument("--slice-min-particles", type=int, default=100)
    parser.add_argument("--slice-grid", type=int, default=1000)
    parser.add_argument("--num-slices", type=int, default=10)
    parser.add_argument("--l-3d", type=float, default=3.0)
    parser.add_argument("--plot-dpi", type=int, default=600)
    parser.add_argument(
        "--line-length-mode",
        choices=["fixed", "uniform"],
        default="fixed",
    )
    parser.add_argument(
        "--density-map",
        choices=["on", "off"],
        default="off",
    )
    parser.add_argument(
        "--line",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--show-all-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mst-multi-scan", action="store_true")
    parser.add_argument(
        "--ortho-view-bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument(
        "--Filament",
        nargs="+",
        help="MST [distance], T-web [threshold], or all [distance] [threshold]",
    )
    parser.add_argument(
        "--thr",
        "--filament-mst-mean-sep-factor",
        dest="filament_mst_mean_sep_factor",
        type=float,
        default=0.3,
    )
    return parser


def get_global_config(
    args: argparse.Namespace,
    data_dir: str | None = None,
) -> GlobalConfig:
    if data_dir is None:
        data_dir = args.data_dir[0] if isinstance(args.data_dir, list) else args.data_dir

    filament_method = None
    filament_mst_dist = -1.0
    filament_tweb_th = 5e-12
    if getattr(args, "Filament", None):
        method = args.Filament[0].lower()
        if method == "mst":
            filament_method = "MST"
            if len(args.Filament) > 1:
                filament_mst_dist = float(args.Filament[1])
        elif method == "t-web":
            filament_method = "T-web"
            if len(args.Filament) > 1:
                filament_tweb_th = float(args.Filament[1])
        elif method == "all":
            filament_method = "all"
            if len(args.Filament) > 1:
                filament_mst_dist = float(args.Filament[1])
            if len(args.Filament) > 2:
                filament_tweb_th = float(args.Filament[2])
        else:
            raise ValueError("--Filament must be MST, T-web, or all")

    return GlobalConfig(
        data_dir=data_dir,
        data_prefix=args.data_prefix,
        output_root=args.output_root,
        num_files=args.nfiles,
        num_workers=min(args.workers, 5),
        random_seed=args.seed,
        box_size=args.box_size,
        grid_size=args.grid_size,
        particle_mass=args.particle_mass,
        min_particles=args.min_particles,
        shape_min_particles=args.shape_min_particles,
        slice_min_particles=args.slice_min_particles,
        slice_grid=args.slice_grid,
        num_slices=args.num_slices,
        l_3d=args.l_3d,
        line_length_mode=args.line_length_mode,
        plot_dpi=args.plot_dpi,
        filament_method=filament_method,
        filament_mst_dist=filament_mst_dist,
        filament_mst_mean_sep_factor=args.filament_mst_mean_sep_factor,
        filament_tweb_th=filament_tweb_th,
        line_only=args.density_map.lower() == "off",
        draw_shape_lines=args.line,
        show_all_mass=args.show_all_mass,
        mst_multi_scan=args.mst_multi_scan,
        ortho_view_bounds=args.ortho_view_bounds,
    )


def get_configs_from_args(args: argparse.Namespace) -> list[GlobalConfig]:
    data_dirs = args.data_dir if isinstance(args.data_dir, list) else [args.data_dir]
    return [get_global_config(args, data_dir) for data_dir in data_dirs]
