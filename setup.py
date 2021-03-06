from setuptools import find_packages, setup

setup(
    name="chinesebert",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.2.1",
    license="MIT",
    description="ChineseBert_pytorch",
    author="Jun Yu",
    author_email="573009727@qq.com",
    url="https://github.com/JunnYu/ChineseBert_pytorch",
    keywords=["ChineseBert", "pytorch"],
    install_requires=["transformers>=4.14.1", "pypinyin==0.38.1"],
)
