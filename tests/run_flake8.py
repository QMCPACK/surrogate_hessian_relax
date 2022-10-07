#!/usr/bin/sh

flake8 --ignore=E251,E221,E262,E265,E501,E722,E711 ../lib/*.py
flake8 --ignore=F401,E251,E265 ../*.py 
