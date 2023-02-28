# poverty_mapper

pyenv install 3.7.7 -- to install 3.7.7  
pyenv virtualenv 3.7.7 poverty-mapper -- to create poverty-mapper venv

## If there's an error on MacOS 11.7

pyenv install --patch 3.7.7 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch) -- to install 3.7.7 
pyenv virtualenv 3.7.7 poverty-mapper -- to create poverty-mapper venv
