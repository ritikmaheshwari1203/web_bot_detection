from setuptools import setup, find_packages

setup(
    name="webbot-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "pandas>=1.2.0",
        "requests>=2.25.0",
    ],
    entry_points={
        "console_scripts": [
            "webbot-detector=webbot_detector.server:cli",
        ],
    },
)