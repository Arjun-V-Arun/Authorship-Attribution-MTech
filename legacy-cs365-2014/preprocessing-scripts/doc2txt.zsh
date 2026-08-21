#!/bin/zsh

for file in *.doc; do
    libreoffice --headless --convert-to txt:text $file
    rm -f $file
done
