
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pLDA',
    version='0.0.1',
    author='Martin McCullagh',
    author_email='martin.mccullagh@okstate.edu',
    description='Linear Discriminant Analysis on Particle Positions',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mccullaghlab/pLDA',
    project_urls = {
        "Bug Tracker": "https://github.com/mccullaghlab/pLDA/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    license='MIT',
    #install_requires=['numpy','torch==1.11','torch_batch_svd'],
    install_requires=['numpy','torch>=1.11', 'shapeGMMTorch', 'sklearn'],
)
