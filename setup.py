from setuptools import setup, find_packages

setup(
    name='mnist_ol',
    version='0.1',
    description='mnist ordered by label',
    url='http://gitlab.com/continual_learning/mnist_ol',
    author='Min Lin',
    author_email='mavenlin@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
    ],
)
