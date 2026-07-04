import os
import numpy as np
from lib.pipeline_config import GlobalConfig


CATALOG_NAME = "0.000_FoF_b0.20_1.bin"
CATALOG_HEADER_DTYPE = np.dtype([
    ("nhalo_tot", "<i4"),
    ("nhalo", "<i4"),
    ("ninfo", "<i4"),
    ("blink", "<f4"),
])
MIN_REQUIRED_NINFO = 98
# Every supplied catalog starts at 20 member particles. This is a measured
# catalog floor, not an accuracy/convergence claim and not an analysis cut.
CATALOG_PARTICLE_FLOOR = 20


def read_catalog_path(file_path: str):
    """Read one catalog after validating its declared schema and exact length."""
    if not os.path.exists(file_path):
        return None

    actual_bytes = os.path.getsize(file_path)
    with open(file_path, "rb") as stream:
        header_bytes = stream.read(CATALOG_HEADER_DTYPE.itemsize)
        header_array = np.frombuffer(header_bytes, dtype=CATALOG_HEADER_DTYPE)
        if len(header_array) != 1:
            raise ValueError(f"catalog header is missing or truncated: {file_path}")

        header = header_array[0]
        nhalo_tot = int(header["nhalo_tot"])
        nhalo = int(header["nhalo"])
        ninfo = int(header["ninfo"])
        if nhalo < 0 or nhalo_tot < nhalo:
            raise ValueError(
                f"invalid halo counts in {file_path}: "
                f"nhalo_tot={nhalo_tot}, nhalo={nhalo}"
            )
        if ninfo < MIN_REQUIRED_NINFO:
            raise ValueError(
                f"catalog ninfo={ninfo} is smaller than the required "
                f"{MIN_REQUIRED_NINFO}: {file_path}"
            )

        expected_bytes = (
            CATALOG_HEADER_DTYPE.itemsize
            + nhalo * ninfo * np.dtype("<f4").itemsize
        )
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"catalog file length mismatch for {file_path}: "
                f"actual={actual_bytes}, expected={expected_bytes}"
            )

        if nhalo == 0:
            return None

        expected_payload_bytes = nhalo * ninfo * 4
        payload_bytes = stream.read(expected_payload_bytes)
        values = np.frombuffer(payload_bytes, dtype="<f4")
        if values.size != nhalo * ninfo:
            raise ValueError(
                f"catalog payload is truncated for {file_path}: "
                f"actual_values={values.size}, expected_values={nhalo * ninfo}"
            )
        data = values.reshape(nhalo, ninfo)

    data.flags.writeable = False
    return data


class CatalogData:
    """
    Read-only adapter that gives the Fortran catalog columns explicit names.

    The adapter preserves backward-compatible numpy-style slicing: existing
    code using data[:, 0] or data[mask, 34:43] continues to receive numpy
    arrays, while data[mask] returns another CatalogData view.
    """

    def __init__(self, raw_data: np.ndarray):
        self.raw = raw_data

    def __len__(self):
        return len(self.raw)

    def __array__(self, dtype=None):
        return np.asarray(self.raw, dtype=dtype)

    def __getitem__(self, key):
        result = self.raw[key]
        if (
            isinstance(result, np.ndarray)
            and result.ndim == 2
            and result.shape[1] == self.raw.shape[1]
        ):
            return CatalogData(result)
        return result

    @property
    def shape(self):
        return self.raw.shape

    @property
    def dtype(self):
        return self.raw.dtype

    @property
    def flags(self):
        return self.raw.flags

    @property
    def particles(self):
        """FoF halo particle count from column 0 of the supplied schema."""
        return self.raw[:, 0]

    @property
    def positions(self):
        """Eulerian halo COM centers in catalog grid coordinates, columns 1:4."""
        return self.raw[:, 1:4]

    @property
    def pos_lagrangian(self):
        """Lagrangian halo COM centers in catalog grid coordinates, columns 7:10."""
        return self.raw[:, 7:10]

    @property
    def xx_flat(self):
        """
        Eulerian COM-centered second-moment tensor, SUMMED over member
        particles, in box-normalized coordinates (position/box length,
        dimensionless).

        Verified against the binary data: trace(xx) carries one additional
        factor of particle count relative to qq, so the particle-averaged
        variance used for RMS axis conversion is xx / Np.

        Absolute-size conversion: an eigenvalue lambda of this matrix maps to
        a principal RMS length of L_box * sqrt(lambda / Np) in Mpc/h.
        """
        return self.raw[:, 34:43]

    @property
    def qq_flat(self):
        """
        Lagrangian COM-centered second-moment tensor, AVERAGED over member
        particles, in box-normalized coordinates (position/box length,
        dimensionless).

        Verified against the binary data: the scalar normalization differs from
        xx by a factor of Np. The qq tensor is interpreted here only as the
        catalog's Lagrangian second-order statistic in q coordinates; it is not
        relabeled as a measured boundary or volume.

        Absolute-size conversion: an eigenvalue lambda of this matrix maps to
        a principal RMS length of L_box * sqrt(lambda) in Mpc/h (no Np factor).
        """
        return self.raw[:, 43:52]

    @property
    def vec_qq(self):
        """
        qq eigenvectors paired by column with catalog columns 70:73.

        The flat catalog block stores each vector contiguously; swapping the
        reshaped row/vector axis exposes the NumPy ``eigh`` column convention.
        """
        return np.swapaxes(self.raw[:, 73:82].reshape(-1, 3, 3), 1, 2)

    @property
    def vec_tide(self):
        """Tidal eigenvectors paired by column with catalog columns 82:85."""
        return np.swapaxes(self.raw[:, 85:94].reshape(-1, 3, 3), 1, 2)

    @property
    def lambda_tide_catalog(self):
        """
        Catalog tidal eigenvalues in columns 82:85.

        The repository diagnostics document these as the physical eigenvalues
        paired with ``vec_tide``; callers that need the named schema should use
        this accessor instead of repeating a raw slice.
        """
        return self.raw[:, 82:85]

UNIFORM_ELLIPSOID_BOUNDARY_FACTOR = np.sqrt(5.0)


def read_catalog_data(fi: int, config: GlobalConfig, custom_path: str = None):
    """
    공통 I/O: 파일을 읽고 파싱하여 열 의미가 명시된 read-only adapter를 반환.
    """
    if custom_path and os.path.isfile(custom_path):
        file_path = custom_path
    else:
        file_path = os.path.join(config.dir_path + str(fi), CATALOG_NAME)

    data = read_catalog_path(file_path)
    if data is None:
        return None
    return CatalogData(data)
