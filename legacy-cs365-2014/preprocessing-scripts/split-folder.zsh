#!/bin/zsh

# Create a new folder for the supplied folder and split the files in it
mkdir ${1}.split
cd ${1}.split

# filter out punctuations, hindi danda and split into 500 word samples
cat ../${1}/*| tr "\n" " " | sed 's/\xe0\xa5\xa4//g' | sed 's/\xe2\x80\x98//g' | sed 's/\xe2\x80\x99//g' \
    | tr "[',?;\")(.!]" " " | sed 's/  */\n/g' | split -l 500

# Move back to the root directory
cd ..
