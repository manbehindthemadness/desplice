from setuptools import setup, find_packages
try:
    from install_preserve import preserve
except ImportError:
    import pip  # noqa
    pip.main(['install', 'install-preserve'])
    from install_preserve import preserve  # noqa

install_requires = [
    'opencv-python',
    'Pillow',
    'numpy',
    'antidupe',
]

exclusions = [
    'opencv-python:cv2'
]

install_requires = preserve(install_requires, exclusions, verbose=True)


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='desplice',
    version='0.0.2',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
        ],
    },
    author='Manbehindthemadness',
    author_email='manbehindthemadness@gmail.com',
    description='A Python library designed to process video files by removing duplicate frames',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/manbehindthemadness/desplice',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
