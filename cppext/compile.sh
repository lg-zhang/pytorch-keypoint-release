#!/usr/bin/env bash
src='process_score.cpp repeatability.cpp'
g++ -fPIC -shared -o libcppext.so $src
