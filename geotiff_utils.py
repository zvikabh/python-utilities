import uuid

import numpy as np
from osgeo import gdal


def read_geotiff(
    filename: str,
) -> tuple[list[np.ma.MaskedArray], str, tuple[float, ...]]:
    """Reads a GeoTIFF, returning all bands and the projection info.

    Args:
      filename: Filename to read.

    Returns:
      image_bands: List of MaskedArrays, one for each band.
      wkt_projection: WKT string for the GeoTIFF projection.
      geotransform: 6-element vector defining the relation between image pixels
        and the coordinate reference system (CRS).
    """
    dataset = gdal.Open(filename)
    return _parse_geotiff_from_gdal_dataset(dataset)


def parse_geotiff(
    data: bytes
) -> tuple[list[np.ma.MaskedArray], str, tuple[float, ...]]:
    """Parses a GeoTIFF which has already been loaded in memory.
    
    This is useful, for example, when reading from a nonstandard file system,
    such as GCS or S3.

    Args:
      data: Encoded GeoTIFF.

    Returns:
      image_bands: List of MaskedArrays, one for each band.
      wkt_projection: WKT string for the GeoTIFF projection.
      geotransform: 6-element vector defining the relation between image pixels
        and the coordinate reference system (CRS).
    """
    mmap_filename = f'/vsimem/{uuid.uuid4().hex}.tif'
    gdal.FileFromMemBuffer(mmap_filename, data)
    dataset = gdal.Open(mmap_filename)
    image_bands, wkt_projection, geotransform = (
        _parse_geotiff_from_gdal_dataset(dataset))
    dataset = None  # Call destructor on the C++ Dataset object.
    gdal.Unlink(mmap_filename)
    return image_bands, wkt_projection, geotransform


def _parse_geotiff_from_gdal_dataset(
    dataset) -> tuple[list[np.ma.MaskedArray], str, tuple[float, ...]]:
    """Parses a GeoTIFF from a GDAL DataSet.
    
    Returns:
      image_bands: List of MaskedArrays, one for each band.
      wkt_projection: WKT string for the GeoTIFF projection.
      geotransform: 6-element vector defining the relation between image pixels
        and the coordinate reference system (CRS).
    """
    wkt_projection = dataset.GetProjection()
    geo_transform = dataset.GetGeoTransform()
    image_bands = []
    for i in range(1, dataset.RasterCount + 1):  # GDAL bands are 1-indexed
        data = dataset.GetRasterBand(i).ReadAsArray()
        mask = dataset.GetRasterBand(i).GetMaskBand().ReadAsArray()
        band = np.ma.MaskedArray(data, ~mask)
        image_bands.append(band)
    return image_bands, wkt_projection, geo_transform


def encode_geotiff(
    data: np.ndarray | np.ma.MaskedArray,
    wkt_projection: str,
    geo_transform: tuple[float, ...]) -> bytearray:
    """Writes a GeoTIFF to a `bytearray` object in memory."""
    mmap_filename = f'/vsimem/{uuid.uuid4().hex}.tif'
    write_geotiff(data, wkt_projection, geo_transform, mmap_filename)
    vsifile = gdal.VSIFOpenL(mmap_filename, 'rb')
    encoded_tiff = gdal.VSIFReadL(1, 1000000000, vsifile)
    gdal.VSIFCloseL(vsifile)
    gdal.Unlink(mmap_filename)
    return encoded_tiff


def write_geotiff(
    data: np.ndarray | np.ma.MaskedArray,
    wkt_projection: str,
    geo_transform: tuple[float, ...],
    filename: str,
) -> None:
    """Writes a GeoTIFF to a file."""
    assert data.ndim == 2
    rows, cols = data.shape
    match data.dtype:
        case np.float64:
            gdal_type = gdal.GDT_Float64
            no_data_value = np.nan
        case np.float32:
            gdal_type = gdal.GDT_Float32
            no_data_value = np.nan
        case np.uint16 | np.uint8:
            gdal_type = gdal.GDT_UInt16
            no_data_value = 65534
        case np.bool_:
            gdal_type = gdal.GDT_Int8
            no_data_value = 64
            data = data.astype(np.int8) * 127
        case _:
            raise KeyError(f'Unknown dtype {data.dtype}')

    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(filename, cols, rows, 1, gdal_type)
    outdata.SetGeoTransform(geo_transform)
    outdata.SetProjection(wkt_projection)
    if isinstance(data, np.ma.MaskedArray):
        outdata.GetRasterBand(1).WriteArray(data.filled(no_data_value))
        outdata.GetRasterBand(1).SetNoDataValue(no_data_value)
    else:
        outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache()


def image_coords_to_geo_coords(
        row: int, col: int, geotransform: tuple[float, ...]
) -> tuple[float, float]:
    """Converts from image coordinates to geographic (CRS) coordinates.
    
    Args:
        row: y-coordinate of the image, where 0 is the topmost row.
        col: x-coordinate of the image, where 0 is the leftmost column.
        geotransform: 6-element geotransform vector; see
            https://gdal.org/tutorials/geotransforms_tut.html
    
    Returns:
        Tuple of coordinates of the center of the specified pixel, in
        geographic (CRS) coordinates.
    """
    # Get coordinates for the pixel center, rather than its top-left corner.
    row += 0.5
    col += 0.5
    gt = np.reshape(geotransform, newshape=(2, 3))
    result = gt @ [1, col, row]
    return result[0], result[1]
