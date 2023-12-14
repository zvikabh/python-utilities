import tempfile
import unittest

import numpy as np

import geotiff_utils

_EPSG_4326_WKT = '''GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'''
_GEOTRANSFORM = (34.5, 0.1, 0.0, 32.9, 0.0, -0.1)


class GeotiffUtilsTest(unittest.TestCase):

    def test_read_geotiff(self):
        bands, wkt, geotransform = geotiff_utils.read_geotiff(
            'testdata/small_tif.tif')
        self.assertEqual(wkt, _EPSG_4326_WKT)
        self.assertEqual(1, len(bands))
        expected_data = np.arange(100).reshape((10, 10)).astype(np.uint8)
        np.testing.assert_array_equal(bands[0], expected_data)
        np.testing.assert_allclose(geotransform, _GEOTRANSFORM)

    def test_read_write_geotiff(self):
        orig_data = np.arange(100).reshape((10, 10)).astype(np.float32)
        with tempfile.NamedTemporaryFile() as f:
            geotiff_utils.write_geotiff(
                orig_data, _EPSG_4326_WKT, _GEOTRANSFORM, f.name)
            loaded_data, loaded_wkt, loaded_geotransform = (
                geotiff_utils.read_geotiff(f.name))
        self.assertEqual(1, len(loaded_data))
        np.testing.assert_allclose(orig_data, loaded_data[0])
        self.assertEqual(loaded_wkt, _EPSG_4326_WKT)
        np.testing.assert_allclose(loaded_geotransform, _GEOTRANSFORM)

    def test_image_coords_to_geo_coords(self):
        np.testing.assert_allclose(
            [34.55, 32.85],
            geotiff_utils.image_coords_to_geo_coords(0, 0, _GEOTRANSFORM))
        np.testing.assert_allclose(
            [34.65, 32.55],
            geotiff_utils.image_coords_to_geo_coords(3, 1, _GEOTRANSFORM))


if __name__ == '__main__':
    unittest.main()
