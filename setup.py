from setuptools import find_packages, setup

setup(
  name="mmrl",
  version="0.1.0",
  description="Environment-agnostic reinforcement learning algorithm package.",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  python_requires=">=3.13",
  package_dir={"": "src"},
  packages=find_packages("src"),
  entry_points={
    "console_scripts": [
      "tdmpc2-train = mmrl.scripts.tdmpc2.train:main",
      "tdmpc2-play = mmrl.scripts.tdmpc2.play:main",
      "fastsac-train = mmrl.scripts.fastsac.train:main",
      "fastsac-play = mmrl.scripts.fastsac.play:main",
    ]
  },
)
