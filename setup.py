
'''
Configure:
python setup.py build

Colin Lea
2016
'''

from distutils.core import setup
from Cython.Distutils import build_ext
import numpy as np

setup(
	author = 'Colin Lea',
	author_email = 'colincsl@gmail.com',
	description = '',
	license = "",
	version= "0.1",
	name = 'Latent Convolutional Timeseries Models',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	packages= [	"LCTM",
				"LCTM.energies",
				],
	# package_data={'':['*.xml', '*.png', '*.yml', '*.txt']},
)
