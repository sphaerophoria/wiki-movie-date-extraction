#!/usr/bin/env bash

curl -o "$1" "https://en.wikipedia.org/w/index.php?title=$1&action=raw"
