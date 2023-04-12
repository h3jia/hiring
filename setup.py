from setuptools import setup, find_packages
import warnings, os, subprocess
from os import path


# from numpy/setup.py, which is distributed under BSD 3-Clause
# https://github.com/numpy/numpy/blob/master/setup.py
# https://github.com/numpy/numpy/blob/master/LICENSE.txt
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='hiring',
    # version='0.1.0.dev1',
    version='0.1.0.dev1+' + git_version()[:7],
    author='He Jia',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    description=('Toolbox for higher order photon rings.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/h3jia/hiring',
    license='Apache License, Version 2.0',
    python_requires=">=3.7",
    install_requires=['jax>=0.3', 'numpy>=1.17', 'scipy'],
)
