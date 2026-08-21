#!/bin/zsh

# Create a new folder for the supplied corpus and split the directories in it
mkdir ${1}.corpus
for folder in ${1}/**(/); do
    echo "Now Processing folder ${folder}"

    # Create a folder for each folder in the corpus
    mkdir ${folder}.split
    cd ${folder}.split

    # prune out hindi danda, quotation marks, and other punctuations and then split into 500 word samples
    cat ../${folder}/*| tr "\n" " " | sed 's/\xe0\xa5\xa4//g' | sed 's/\xe2\x80\x98//g' | sed 's/\xe2\x80\x99//g' \
        | tr "[',?;\")(.!]" " " | sed 's/  */\n/g' | split -l 500

    # cd back to the root corpus directory
    cd ..
    mv ${folder}.split ${1}.corpus
done
