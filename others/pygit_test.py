# coding: utf-8

import sys

from git import Repo


repo = Repo('../')
print(repo.commit('master'))
