from setuptools import setup, find_packages

setup(
    name='bitcoin_analysis',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.3',
        'numpy==1.26.4',
    ],
    py_modules=['bitcoin_analysis'],
) 