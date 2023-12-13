import unittest

import numpy as np

import image_utils


class ImageUtilsTest(unittest.TestCase):

    def test_read_image(self):
        img = image_utils.read_image('testdata/small_img.png')
        expected_img = np.asarray(
            [[0, 25, 51], [76, 102, 127], [153, 178, 204], [229, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(img, expected_img)

    def test_encode_then_decode(self):
        img = np.arange(20).reshape((4, 5)) / 20
        encoded_img = image_utils.encode_image(img, '.png')
        decoded_img = image_utils.decode_image(encoded_img)
        np.testing.assert_allclose(decoded_img, (img * 255).astype(np.uint8))


if __name__ == '__main__':
    unittest.main()
