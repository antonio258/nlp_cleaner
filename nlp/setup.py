import os
import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

requeriments = f'{os.path.dirname(os.path.realpath(__file__))}/requirements.txt'

if os.path.isfile(requeriments):
    with open(requeriments) as f:
        install_requires = f.read().splitlines()

setup(
    name = "nlp",
    version = "1.0.0",
    author = "Antônio Pereira",
    author_email = "antonio258p@gmail.com",
    description = ("NLP Processing"),
    license = "",
    keywords = "nlp processing text",
    url = "",
    packages=find_packages(exclude='docs'),
    entry_points='''
        [console_scripts]
    ''',
    install_requires=install_requires,
)