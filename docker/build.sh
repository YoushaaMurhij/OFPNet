#!/bin/bash

docker build . \
             -f Dockerfile \
             -t x64/ofp:latest 