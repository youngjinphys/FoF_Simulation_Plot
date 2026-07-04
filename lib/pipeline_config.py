import os
import tempfile
import math

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib_cache"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
from dataclasses import dataclass
from numbers import Integral, Real


def numeric_token(value: float, min_decimals: int = 0) -> str:
    """Compact, collision-resistant numeric token for output/cache paths."""
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
    """
    전체 파이프라인 중앙 설정 클래스.
    모든 스크립트에서 공통으로 사용되는 경로 및 물리 상수 관리.
    """
    data_dir: str = "./data/FoF_R"          # ./data/FoF_E 또는 ./data/FoF_R
    data_prefix: str = "resR"          # resE 또는 resR
    output_root: str = "./output"      # 결과 출력 루트
    
    num_files: int = 500
    num_workers: int = 5
    
    box_size: float = 100.0
    grid_size: int = 256
    particle_mass: float = 4.27e9
    min_particles: int = 20
    shape_min_particles: int = 100

    # spatial_slices_projection.py
    slice_min_particles: int = 100
    slice_grid: int = 1000
    num_slices: int = 10
    l_3d: float = 3.0
    line_length_mode: str = "fixed"
    plot_dpi: int = 600

    # CosmicWeb_Classification.py
    cw_mass_threshold: float = 1e12
    cw_logm_min: float = 11.5
    cw_logm_max: float = 15.0
    cw_num_bins: int = 30
    cw_fiducial_th: float = 5e-12

    # VoidFinder_Profile.py
    void_min_particles: int = 10
    void_grid_size: int = 32
    void_smooth_mpc: float = 5.0
    void_delta_void: float = -0.7
    void_delta_edge: float = -0.2
    void_r_min: float = 8.0
    void_r_max: float = 30.0
    void_max_per_file: int = 50
    void_profile_bins: int = 20
    void_r_over_rv_max: float = 3.0

    # two_point_cf.py
    tpcf_r_min: float = 0.5
    tpcf_r_max: float = 173.3
    tpcf_log_bin_step: float = 0.05

    # VectorChirality_PS.py
    vc_grid_size: int = 32
    vc_num_bins: int = 10

    # SpinAlignment_Analysis.py
    spin_min_particles: int = 100
    spin_lambda_th: float = 0.0
    spin_logm_min: float = 11.5
    spin_logm_max: float = 14.5
    spin_num_bins: int = 11
    spin_pdf_mass_bin_step: float = 0.25
    spin_low_logm_max: float = 12.0
    spin_high_logm_min: float = 13.0
    spin_shape_low_logm_max: float = 12.5
    spin_shape_high_logm_min: float = 13.0
    spin_min_per_file_bin_count: int = 1
    spin_min_files_per_bin_for_plot: int = 1

    # IntrinsicAlignment_IA.py
    ia_r_min: float = 1.0
    ia_r_max: float = 30.0
    ia_num_bins: int = 15
    ia_chunk_size: int = 500

    # TTT_Verification.py
    ttt_logm_min: float = 11.5
    ttt_logm_max: float = 14.5
    ttt_num_bins: int = 14

    # halo_shape.py
    shape_logm_min: float = 11.0
    shape_logm_max: float = 15.5
    shape_logm_bin_step: float = 0.15

    # HMF_Plot.py
    hmf_log_bin_step: float = 0.1
    hmf_logm_min: float = 10.0
    hmf_logm_max: float = 16.0

    # Visualization.py
    vis_crop_min: float = 20.0
    vis_crop_max: float = 60.0
    vis_realization: int = 0
    cosmology_name: str = "Planck"
    redshift: float = 0.0
    random_seed: int = 42
    
    # Filament Options
    filament_method: str = None  # "MST", "T-web", or "all"
    filament_mst_dist: float = -1.0
    filament_mst_mean_sep_factor: float = 0.3
    filament_tweb_th: float = 5e-12
    line_only: bool = True
    draw_shape_lines: bool = False
    show_all_mass: bool = True
    mst_multi_scan: bool = False
    
    # Orthographic View Option
    ortho_view_bounds: tuple = None  # (xmin, xmax, ymin, ymax, zmin, zmax)

    def __post_init__(self):
        """Normalize paths and reject settings that cannot yield valid results."""
        if isinstance(self.data_dir, list):
            if not self.data_dir:
                raise ValueError("data_dir list cannot be empty")
            self.data_dir = self.data_dir[0]

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

        for name in ("data_dir", "data_prefix", "output_root", "cosmology_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("box_size", "particle_mass", "l_3d"):
            require_positive_real(name)
        for name in (
            "grid_size", "num_files", "num_workers", "slice_grid",
            "num_slices", "plot_dpi", "cw_num_bins", "void_grid_size",
            "void_max_per_file", "void_profile_bins", "vc_grid_size",
            "vc_num_bins", "spin_num_bins", "spin_min_per_file_bin_count",
            "spin_min_files_per_bin_for_plot", "ia_num_bins", "ia_chunk_size",
            "ttt_num_bins",
        ):
            require_positive_int(name)
        for name in (
            "min_particles", "shape_min_particles", "slice_min_particles",
            "void_min_particles", "spin_min_particles",
        ):
            require_nonnegative_int(name)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, Integral):
            raise ValueError("random_seed must be an integer")
        if not isinstance(self.redshift, Real) or not math.isfinite(self.redshift):
            raise ValueError("redshift must be finite")
        if self.line_length_mode not in {"fixed", "uniform"}:
            raise ValueError("line_length_mode must be 'fixed' or 'uniform'")
        if self.filament_method not in {"MST", "T-web", "all", None}:
            raise ValueError("filament_method must be 'MST', 'T-web', 'all', or None")
        require_positive_real("filament_mst_mean_sep_factor")
        if not isinstance(self.filament_mst_dist, Real) or not math.isfinite(self.filament_mst_dist):
            raise ValueError("filament_mst_dist must be finite")
        if self.filament_mst_dist == 0.0:
            raise ValueError("filament_mst_dist must be negative for auto mode or positive")

        for prefix, step_name in (
            ("hmf", "hmf_log_bin_step"),
            ("shape", "shape_logm_bin_step"),
        ):
            lower = getattr(self, f"{prefix}_logm_min")
            upper = getattr(self, f"{prefix}_logm_max")
            if not all(isinstance(value, Real) and math.isfinite(value) for value in (lower, upper)):
                raise ValueError(f"{prefix} log-mass bounds must be finite")
            require_positive_real(step_name)
            if upper <= lower:
                raise ValueError(f"{prefix}_logm_max must be greater than {prefix}_logm_min")
            intervals = (upper - lower) / getattr(self, step_name)
            if not math.isclose(intervals, round(intervals), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(
                    f"{prefix}_logm_max - {prefix}_logm_min must be an integer multiple of {step_name}"
                )

        for prefix in ("cw", "spin", "ttt"):
            lower = getattr(self, f"{prefix}_logm_min")
            upper = getattr(self, f"{prefix}_logm_max")
            if not all(isinstance(value, Real) and math.isfinite(value) for value in (lower, upper)):
                raise ValueError(f"{prefix} log-mass bounds must be finite")
            if upper <= lower:
                raise ValueError(f"{prefix}_logm_max must be greater than {prefix}_logm_min")

        for name in (
            "cw_mass_threshold", "void_smooth_mpc", "void_r_min",
            "void_r_max", "void_r_over_rv_max", "ia_r_min", "ia_r_max",
        ):
            require_positive_real(name)
        for name in ("cw_fiducial_th", "spin_lambda_th", "void_delta_void", "void_delta_edge"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.void_r_max <= self.void_r_min:
            raise ValueError("void_r_max must be greater than void_r_min")
        if self.void_delta_edge <= self.void_delta_void:
            raise ValueError("void_delta_edge must be greater than void_delta_void")
        if self.ia_r_max <= self.ia_r_min:
            raise ValueError("ia_r_max must be greater than ia_r_min")

        if not isinstance(self.data_dir, str):
            raise ValueError("data_dir must be a path string or non-empty list")
        pdf_intervals = (
            (self.spin_logm_max - self.spin_logm_min)
            / self.spin_pdf_mass_bin_step
            if self.spin_pdf_mass_bin_step > 0.0
            else math.nan
        )
        if (
            not math.isfinite(self.spin_pdf_mass_bin_step)
            or self.spin_pdf_mass_bin_step <= 0.0
            or not math.isfinite(pdf_intervals)
            or not math.isclose(pdf_intervals, round(pdf_intervals), abs_tol=1.0e-10)
        ):
            raise ValueError(
                "spin_pdf_mass_bin_step must be positive and exactly divide "
                "spin_logm_max - spin_logm_min"
            )

        if "FoF_E" in self.data_dir and self.data_prefix == "resR":
            self.data_prefix = "resE"
        elif "FoF_R" in self.data_dir and self.data_prefix == "resE":
            self.data_prefix = "resR"
        
        # Ensure data_dir doesn't end with slash
        self.data_dir = self.data_dir.rstrip('/')

    @property
    def coord_scale(self) -> float:
        """격자 단위를 물리 단위(Mpc/h)로 변환하는 스케일 팩터"""
        return self.box_size / self.grid_size

    @property
    def particle_implied_omega_m(self) -> float:
        """
        Matter density implied if the box contains grid_size**3 equal particles.

        This constrains Omega_m but not H0, sigma8, ns, or the full simulation
        cosmology, so it is a consistency diagnostic rather than a replacement
        for simulation metadata.
        """
        rho_crit_h2 = 2.775e11
        rho_m = self.particle_mass * self.grid_size**3 / self.box_size**3
        return rho_m / rho_crit_h2
    
    @property
    def dir_path(self) -> str:
        """데이터 파일이 있는 디렉토리 경로"""
        return f"{self.data_dir}/{self.data_prefix}"
    
    def output_dir(self, task_name: str) -> str:
        """작업(task)별 출력 디렉토리 반환 (데이터셋 이름별로 폴더 분리하여 덮어쓰기 방지, 필요시 생성)"""
        dataset_label = os.path.basename(self.data_dir.rstrip('/'))
        d = os.path.join(self.output_root, dataset_label, task_name)
        os.makedirs(d, exist_ok=True)
        return d
    
    def output_filename(self, plot_name: str, ext: str = "png") -> str:
        """데이터셋 이름이 포함된 표준화된 출력 파일명 반환"""
        dataset_label = os.path.basename(self.data_dir.rstrip('/'))
        return f"{plot_name}_{dataset_label}.{ext}"


def setup_common_argparse(description: str) -> argparse.ArgumentParser:
    """공통 argparse 설정 반환"""
    parser = argparse.ArgumentParser(description=description)
    # 주의: nargs='+'로 인해 args.data_dir은 항상 리스트를 반환합니다.
    # 기존 단일 디렉토리를 처리하던 12개 이상의 스크립트들은 모두 get_global_config()를 사용하며,
    # 해당 함수 내부에서 args.data_dir[0] if isinstance(list) else args.data_dir로 처리하므로 
    # 하위 호환성이 안전하게 유지됩니다.
    parser.add_argument("--data-dir", type=str, nargs='+', default=["./data/FoF_R"], 
                        help="Data directories (e.g., ./data/FoF_R ./data/FoF_E)")
    parser.add_argument("--data-prefix", type=str, default="resR", 
                        help="Data prefix (e.g., resR or resE)")
    parser.add_argument("--output-root", type=str, default="./output", 
                        help="Root output directory")
    parser.add_argument("--nfiles", type=int, default=500, 
                        help="Number of files to process")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of multiprocessing workers")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Global random seed for reproducible sampling")
    parser.add_argument(
        "--dry-run", "--dryrun",
        dest="dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Validate a seeded random realization per input directory by comparing "
            "direct, chunked, and multiprocessing reductions; print a PASS/FAIL report and exit"
        ),
    )

    # 추가 파라미터들
    parser.add_argument("--box-size", type=float, default=100.0, help="Simulation box size in Mpc/h")
    parser.add_argument("--grid-size", type=int, default=256, help="Simulation grid size")
    parser.add_argument("--particle-mass", type=float, default=4.27e9, help="Simulation particle mass")
    parser.add_argument("--min-particles", type=int, default=20, help="Minimum particles for general catalog")
    parser.add_argument("--shape-min-particles", type=int, default=100, help="Minimum particles for shape catalog")
    parser.add_argument("--cosmology-name", type=str, default="Planck", help="Cosmology name for colossus (e.g., Planck, WMAP9)")
    parser.add_argument("--redshift", type=float, default=0.0, help="Redshift of the simulation snapshot")

    # spatial_slices_projection.py
    parser.add_argument("--slice-min-particles", type=int, default=100, help="Min particles for spatial slices")
    parser.add_argument("--slice-grid", type=int, default=1000, help="Splatting grid size")
    parser.add_argument("--num-slices", type=int, default=10, help="Number of projection slices")
    parser.add_argument("--l-3d", type=float, default=3.0, help="Fixed full line length in Mpc/h for spatial slices")
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=600,
        help="DPI for mass-produced spatial-slice plots",
    )
    parser.add_argument(
        "--line-length-mode",
        type=str,
        choices=["fixed", "uniform"],
        default="fixed",
        help=(
            "Major-axis line length when --line is enabled. fixed uses --l-3d; "
            "uniform converts the projected second-moment tensor to a uniform-ellipsoid silhouette length."
        ),
    )
    parser.add_argument("--density-map", type=str, choices=["on", "off"], default="off", help="Enable/disable density map background (on: splatting, off: white background)")
    parser.add_argument(
        "--line",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw fixed-length projected halo major-axis lines.",
    )
    parser.add_argument(
        "--show-all-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use total halo mass for the density background (default). "
            "Use --no-show-all-mass for the filament-probability-weighted map."
        ),
    )
    parser.add_argument("--mst-multi-scan", action='store_true', help="Run MST threshold scans [0.5, 3.0] in addition to dynamic mean_sep")
    parser.add_argument("--ortho-view-bounds", nargs=6, type=float, default=None, metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'), help="Enable 2x2 orthographic projection for the specified 3D bounding box")
    
    # Filament option
    parser.add_argument("--Filament", nargs='+', help="Method to identify/draw filaments. 'MST [dist]', 'T-web [th]', or 'all [dist] [th]'")
    parser.add_argument(
        "--thr",
        "--filament-mst-mean-sep-factor",
        dest="filament_mst_mean_sep_factor",
        type=float,
        default=0.3,
        help="Mean-separation multiplier for automatic MST linking distance; default 0.3",
    )
    
    # CosmicWeb_Classification.py
    parser.add_argument("--cw-mass-threshold", type=float, default=1e12, help="Mass threshold for spin analysis")
    parser.add_argument("--cw-logm-min", type=float, default=11.5, help="Min logM for CosmicWeb HMF")
    parser.add_argument("--cw-logm-max", type=float, default=15.0, help="Max logM for CosmicWeb HMF")
    parser.add_argument("--cw-num-bins", type=int, default=30, help="Number of mass bins for CosmicWeb HMF")
    parser.add_argument("--cw-fiducial-th", type=float, default=5e-12, help="Fiducial threshold for Hahn classification")

    # VoidFinder_Profile.py
    parser.add_argument("--void-min-particles", type=int, default=10, help="Min particles for voids")
    parser.add_argument("--void-grid-size", type=int, default=32, help="Grid size for density field")
    parser.add_argument("--void-smooth-mpc", type=float, default=5.0, help="Gaussian smoothing scale in Mpc/h")
    parser.add_argument("--void-delta-void", type=float, default=-0.7, help="Seed threshold density contrast")
    parser.add_argument("--void-delta-edge", type=float, default=-0.2, help="Void edge density contrast")
    parser.add_argument("--void-r-min", type=float, default=8.0, help="Min void radius in Mpc/h")
    parser.add_argument("--void-r-max", type=float, default=30.0, help="Max void radius in Mpc/h")
    parser.add_argument("--void-max-per-file", type=int, default=50, help="Max voids to process per realization file")
    parser.add_argument("--void-profile-bins", type=int, default=20, help="Number of profile bins")
    parser.add_argument("--void-r-over-rv-max", type=float, default=3.0, help="Max r/R_v for stacked profile")

    # two_point_cf.py
    parser.add_argument("--tpcf-r-min", type=float, default=0.5, help="Min separation for 2pCF in Mpc/h")
    parser.add_argument("--tpcf-r-max", type=float, default=173.3, help="Max separation for 2pCF in Mpc/h")
    parser.add_argument("--tpcf-log-bin-step", type=float, default=0.05, help="Log step for 2pCF binning")

    # VectorChirality_PS.py
    parser.add_argument("--vc-grid-size", type=int, default=32, help="Grid size for vector field CIC")
    parser.add_argument("--vc-num-bins", type=int, default=10, help="Number of wavemode bins")

    # SpinAlignment_Analysis.py
    parser.add_argument(
        "--spin-min-particles",
        "--shape-alignment-min-particles",
        dest="spin_min_particles",
        type=int,
        default=100,
        help=(
            "Minimum particles for the shape-tidal alignment subset. "
            "Spin-direction statistics use --min-particles."
        ),
    )
    parser.add_argument("--spin-lambda-th", type=float, default=0.0, help="Fiducial threshold for spin-tide alignment")
    parser.add_argument("--spin-logm-min", type=float, default=11.5, help="Min logM for spin-tide bins")
    parser.add_argument("--spin-logm-max", type=float, default=14.5, help="Max logM for spin-tide bins")
    parser.add_argument("--spin-num-bins", type=int, default=11, help="Number of mass bins for spin-tide alignment")
    parser.add_argument(
        "--spin-pdf-mass-bin-step",
        type=float,
        default=0.25,
        help="Log10 halo-mass interval for fine mass-stratified alignment PDFs",
    )
    parser.add_argument("--spin-min-per-file-bin-count", type=int, default=1, help="Min halos per mass bin per file")
    parser.add_argument("--spin-min-files-per-bin-for-plot", type=int, default=1, help="Min valid files required per bin")
    parser.add_argument("--spin-low-logm-max", type=float, default=12.0, help="Max mass for low-mass spin PDF")
    parser.add_argument("--spin-high-logm-min", type=float, default=13.0, help="Min mass for high-mass spin PDF")
    parser.add_argument("--spin-shape-low-logm-max", type=float, default=12.5, help="Max mass for low-mass shape PDF")
    parser.add_argument("--spin-shape-high-logm-min", type=float, default=13.0, help="Min mass for high-mass shape PDF")

    # IntrinsicAlignment_IA.py
    parser.add_argument("--ia-r-min", type=float, default=1.0, help="Min rp for IA ED correlation")
    parser.add_argument("--ia-r-max", type=float, default=30.0, help="Max rp for IA ED correlation")
    parser.add_argument("--ia-num-bins", type=int, default=15, help="Number of bins for IA")
    parser.add_argument("--ia-chunk-size", type=int, default=500, help="Chunk size for IA pair search")

    # TTT_Verification.py
    parser.add_argument("--ttt-logm-min", type=float, default=11.5, help="Min logM for TTT bins")
    parser.add_argument("--ttt-logm-max", type=float, default=14.5, help="Max logM for TTT bins")
    parser.add_argument("--ttt-num-bins", type=int, default=14, help="Number of mass bins for TTT analysis")

    # halo_shape.py
    parser.add_argument("--shape-logm-min", type=float, default=11.0, help="Min logM for shape plot bounds")
    parser.add_argument("--shape-logm-max", type=float, default=15.5, help="Max logM for shape plot bounds")
    parser.add_argument("--shape-logm-bin-step", type=float, default=0.15, help="Bin step (dex) for shape logM bins; smaller = finer mass resolution")

    # HMF_Plot.py
    parser.add_argument("--hmf-log-bin-step", type=float, default=0.1, help="Log step for HMF bins")
    parser.add_argument("--hmf-logm-min", type=float, default=10.0, help="Min logM for HMF bins")
    parser.add_argument("--hmf-logm-max", type=float, default=16.0, help="Max logM for HMF bins")

    # Visualization.py
    parser.add_argument("--vis-crop-min", type=float, default=20.0, help="Min crop range in Mpc/h")
    parser.add_argument("--vis-crop-max", type=float, default=60.0, help="Max crop range in Mpc/h")
    parser.add_argument("--vis-realization", type=int, default=0, help="Realization index for 3D visualization")

    return parser


def get_global_config(args: argparse.Namespace, data_dir: str = None) -> GlobalConfig:
    """argparse 결과로부터 GlobalConfig 생성"""
    if data_dir is None:
        data_dir = args.data_dir[0] if isinstance(args.data_dir, list) else args.data_dir
        
    filament_method = None
    filament_mst_dist = -1.0
    filament_tweb_th = 5e-12
    
    if hasattr(args, 'Filament') and args.Filament:
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
                
    line_only = getattr(args, 'density_map', 'off').lower() == 'off'
        
    return GlobalConfig(
        data_dir=data_dir,
        data_prefix=args.data_prefix,
        output_root=args.output_root,
        num_files=args.nfiles,
        num_workers=min(args.workers, 5),
        random_seed=args.seed,
        filament_method=filament_method,
        filament_mst_dist=filament_mst_dist,
        filament_mst_mean_sep_factor=getattr(args, 'filament_mst_mean_sep_factor', 0.3),
        filament_tweb_th=filament_tweb_th,
        line_only=line_only,
        draw_shape_lines=getattr(args, 'line', False),
        show_all_mass=getattr(args, 'show_all_mass', False),
        ortho_view_bounds=getattr(args, 'ortho_view_bounds', None),
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
        # CosmicWeb_Classification.py
        cw_mass_threshold=args.cw_mass_threshold,
        cw_logm_min=args.cw_logm_min,
        cw_logm_max=args.cw_logm_max,
        cw_num_bins=args.cw_num_bins,
        cw_fiducial_th=args.cw_fiducial_th,
        void_min_particles=args.void_min_particles,
        void_grid_size=args.void_grid_size,
        void_smooth_mpc=args.void_smooth_mpc,
        void_delta_void=args.void_delta_void,
        void_delta_edge=args.void_delta_edge,
        void_r_min=args.void_r_min,
        void_r_max=args.void_r_max,
        void_max_per_file=args.void_max_per_file,
        void_profile_bins=args.void_profile_bins,
        void_r_over_rv_max=args.void_r_over_rv_max,
        tpcf_r_min=args.tpcf_r_min,
        tpcf_r_max=args.tpcf_r_max,
        tpcf_log_bin_step=args.tpcf_log_bin_step,
        vc_grid_size=args.vc_grid_size,
        vc_num_bins=args.vc_num_bins,
        spin_min_particles=args.spin_min_particles,
        spin_lambda_th=args.spin_lambda_th,
        spin_logm_min=args.spin_logm_min,
        spin_logm_max=args.spin_logm_max,
        spin_num_bins=args.spin_num_bins,
        spin_pdf_mass_bin_step=args.spin_pdf_mass_bin_step,
        spin_low_logm_max=args.spin_low_logm_max,
        spin_high_logm_min=args.spin_high_logm_min,
        spin_shape_low_logm_max=args.spin_shape_low_logm_max,
        spin_shape_high_logm_min=args.spin_shape_high_logm_min,
        spin_min_per_file_bin_count=args.spin_min_per_file_bin_count,
        spin_min_files_per_bin_for_plot=args.spin_min_files_per_bin_for_plot,
        ia_r_min=args.ia_r_min,
        ia_r_max=args.ia_r_max,
        ia_num_bins=args.ia_num_bins,
        ia_chunk_size=args.ia_chunk_size,
        ttt_logm_min=args.ttt_logm_min,
        ttt_logm_max=args.ttt_logm_max,
        ttt_num_bins=args.ttt_num_bins,
        shape_logm_min=args.shape_logm_min,
        shape_logm_max=args.shape_logm_max,
        shape_logm_bin_step=args.shape_logm_bin_step,
        hmf_log_bin_step=args.hmf_log_bin_step,
        hmf_logm_min=args.hmf_logm_min,
        hmf_logm_max=args.hmf_logm_max,
        vis_crop_min=args.vis_crop_min,
        vis_crop_max=args.vis_crop_max,
        vis_realization=args.vis_realization,
        cosmology_name=args.cosmology_name,
        redshift=args.redshift
    )


def get_configs_from_args(args: argparse.Namespace) -> list[GlobalConfig]:
    """다중 입력 폴더에 대한 config 리스트 반환"""
    data_dirs = args.data_dir if isinstance(args.data_dir, list) else [args.data_dir]
    return [get_global_config(args, ddir) for ddir in data_dirs]
