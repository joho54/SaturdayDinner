#!/usr/bin/env python3
"""
Saturday Dinner 패키지 설치 스크립트
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt 파일 읽기
def read_requirements():
    requirements = []
    if os.path.exists("requirements_improved.txt"):
        with open("requirements_improved.txt", "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="saturday-dinner",
    version="1.0.0",
    author="Saturday Dinner Team",
    author_email="",
    description="수어 인식 및 처리를 위한 통합 라이브러리",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/saturday-dinner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "saturday_dinner": [
            "data/*.csv",
            "data/*.json",
            "specs/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "saturday-quiz=saturday_dinner.core.sign_quiz:main",
            "saturday-train=saturday_dinner.core.main:main",
        ],
    },
    zip_safe=False,
) 