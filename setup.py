from setuptools import setup, find_packages

setup(
    name='d4d',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'd4d-train = deep4downscaling.console.main_train:main',
            'd4d-predict = deep4downscaling.console.main_predict:main',
            'd4d-datasets-inspect = deep4downscaling.console.main_inspect:main',
            'd4d-datasets-create = deep4downscaling.console.main_create:main'
        ]
    }
)

