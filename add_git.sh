#!/bin/bash

# Loop over everything in the current folder (non-hidden items)
for item in *; do
    git add "$item"
done
