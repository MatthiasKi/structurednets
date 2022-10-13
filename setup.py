import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="structurednets",
    version="0.0.2",
    author="Matthias Kissel",
    author_email="matthias.kissel@tum.de",
    description="Algorithms for using structured weight matrices in neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasKi/structurednets",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=['nose'],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        "scipy",
        "torch",
	"torchvision",
        "numpy",
        "pyfaust",
    	"Pillow",
    	"scikit-learn",
    	"tvsclib @ git+https://github.com/MatthiasKi/tvsclib",
    ],
)
