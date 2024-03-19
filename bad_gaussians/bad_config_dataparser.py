"""
BAD-Gaussians dataparser configs.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from bad_gaussians.image_restoration_dataparser import ImageRestorationDataParserConfig
from bad_gaussians.deblur_nerf_dataparser import DeblurNerfDataParserConfig

ImageRestoreDataParser = DataParserSpecification(config=ImageRestorationDataParserConfig())
DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
