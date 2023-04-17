from setuptools import setup, find_packages

# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kiv-nlp-cross-lingual-transformations",
    install_requires=['torch',
                       'numpy',
                        'gensim==3.6.0'],
    version="0.5.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Cross-lingual transformations library",
    license='MIT',
    author='Pavel Priban',
    author_email = 'pribanp@kiv.zcu.cz, amistera@kiv.zcu.cz',
    url = 'https://github.com/pauli31/cross-lingual-transform-method/src/master/',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # packages='kiv-nlp-cross-lingual-transformations',
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
      ]
)