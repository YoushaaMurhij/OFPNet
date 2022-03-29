#!/bin/bash

docker build . \
             -f Dockerfile \
             -t x64/occupancy_flow:latest 