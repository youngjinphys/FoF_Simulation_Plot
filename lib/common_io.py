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
    Read-only adapter that gives the catalog columns used by this visualizer
    explicit names while preserving NumPy-style slicing.
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
    def xx_flat(self):
        """
        Eulerian COM-centered second-moment tensor, summed over member
        particles, in box-normalized coordinates.
        """
        return self.raw[:, 34:43]

    @property
    def qq_flat(self):
        """
        Lagrangian COM-centered particle-averaged second-moment tensor in
        box-normalized coordinates.
        """
        return self.raw[:, 43:52]


def read_catalog_data(fi: int, config: GlobalConfig, custom_path: str = None):
    """Read one configured or explicitly supplied catalog into CatalogData."""
    if custom_path and os.path.isfile(custom_path):
        file_path = custom_path
    else:
        file_path = os.path.join(config.dir_path + str(fi), CATALOG_NAME)

    data = read_catalog_path(file_path)
    if data is None:
        return None
    return CatalogData(data)
