from setuptools import setup

setup(
    name='mahjong_drawer',
    version='0.7.0',
    packages=['mahjong_drawer'],
    entry_points={
        'console_scripts': [
            'mdraw = mahjong_drawer.__main__:main'
        ]
    })
