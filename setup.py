from setuptools import setup

setup(
    name='doenut',
    version='0.1.0',
    description='Design Of Experiments Nice User Toolkit',
    url='https://github.com/ellagale/doenut',
    author='Ella Gale',
    author_email='ella.gale@bristol.ac.uk',
    license='GPL3',
    packages=['doenut'],
    install_requires=['numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering'
    ],
)
