from setuptools import setup, find_packages

setup(
    name='bitcoin_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.3',
        'numpy==1.26.4',
        'arch==5.3.1',
    ],
    py_modules=['bitcoin_analysis'],
) 