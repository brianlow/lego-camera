#!/bin/bash

# change to your directory
cd $1

for file in *.jpeg
do
  # use regular expression to capture the parts
  if [[ $file =~ ^([0-9]+)-(.+)\.jpeg$ ]]; then
    # extract the parts
    number="${BASH_REMATCH[1]}"
    base="${BASH_REMATCH[2]}"
    # rename the file
    mv -- "$file" "${base}.${number}.jpeg"
  fi
done
