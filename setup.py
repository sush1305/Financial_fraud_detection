from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="fraud-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning system for detecting financial fraud",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-fraud-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.12b0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "pytest-cov>=3.0.0",
        ],
        "streaming": [
            "kafka-python>=2.0.2",
            "pyspark>=3.2.0",
        ],
        "dashboard": [
            "dash>=2.0.0",
            "dash-bootstrap-components>=1.0.0",
            "plotly>=5.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
