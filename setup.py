from setuptools import find_packages, setup

setup(
    name="glycebert",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.3",
    license="MIT",
    description="GlyceBert_pytorch",
    author="Jun Yu",
    author_email="573009727@qq.com",
    url="https://github.com/JunnYu/GlyceBert_pytorch",
    keywords=["GlyceBert", "pytorch"],
    install_requires=["transformers>=4.7.0", "pypinyin"],
)
