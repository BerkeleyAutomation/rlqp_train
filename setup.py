import setuptools


# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project
setuptools.setup(
    name="rlqp_train",
    version="1.0.0",
    author="anonymous",
    author_email="anonymous",
    description="RLQP training code and scripts",
    install_requires=[
        'torch>=1.8.1',
        'tensorboard',
        'schema',
        'pandas' # needed by benchmarks
    ],
    packages=["rlqp_train"]
)
