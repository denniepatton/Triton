from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='Triton',
    version="1.0.0",
    author='Robert (Dennie) Patton',
    author_email='rpatton@fredhutch.org',
    description='Testing installation of Triton Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/denniepatton/Triton',
    license='BSD',
    packages=['Triton'],
    install_requires=['requests'],
)