# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Soccer Analytics Final Project',
    version='0.1.0',
    description='Creating basic models used in modern football/soccer analytics',
    long_description=readme,
    author='Casey Carr',
    author_email='ccarr2017@gmail.com',
    url='https://github.com/CarrC2021/SoccerAnalytics',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
