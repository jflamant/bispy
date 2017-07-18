#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
	# If in OSX, we need to install python by hand.
	# We do that using homebrew, pyenv and pyenv-virtualenv
	# You should normally not change anything in here
    brew update >/dev/null
    brew outdated pyenv || brew upgrade --quiet pyenv
    brew install homebrew/boneyard/pyenv-pip-rehash
    brew install pyenv
    eval "$(pyenv init -)"

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv

    # See all available PYTHON versions with `pyenv install --list'.
    pyenv install ${PYTHON}
    pyenv global ${PYTHON}

    export PYENV_VERSION=${PYTHON}
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv-virtualenv venv
    source venv/bin/activate
else
	# Additional installation instructions for UNIX
    # sudo apt-get install -qq gcc g++
fi