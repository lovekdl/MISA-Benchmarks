from setuptools import setup, find_packages

setup(
    name='BCD_Benchmarks',
    version='1.0.0',
    
    author='Thomas ROBERT',
    author_email='thomas.robert.x21@polytechnique.edu',

    description='BCD Finetuning Benchmarks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    # url='https://github.com/IST-DASLab/LDAdam',

    packages=find_packages(exclude=['experiments']),

    python_requires='>=3.9',
    install_requires=["torch>=2.0.0"]
)