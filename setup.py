from setuptools import find_packages, setup

setup(
  name="mjlab-algo",
  version="0.1.0",
  description="Private MJLab extension package for additional RL algorithms.",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  python_requires=">=3.13",
  package_dir={"": "src"},
  packages=find_packages("src"),
  entry_points={
    "console_scripts": [
      "tdmpc2-train = mjlab_algo.scripts.tdmpc2.train:main",
      "tdmpc2-play = mjlab_algo.scripts.tdmpc2.play:main",
      "fastsac-train = mjlab_algo.scripts.fastsac.train:main",
      "fastsac-play = mjlab_algo.scripts.fastsac.play:main",
    ]
  },
)
