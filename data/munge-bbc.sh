#!/usr/bin/env bash

# Given the documents extracted from bbc-fulltext.zip...

echo "Munging $1 files..."

pushd "bbc/$1"

for f in *.txt; do
    line_file="${f/.txt/.line}"
    printf '"' > "$line_file"
    cat --squeeze-blank "$f" |
        tr '\n' ' ' |
        tr '"' "'" >> "$line_file"
    printf '"\n' >> "$line_file"
done

echo 'text' > "../$1.csv"
cat --squeeze-blank *.line >> "../$1.csv"

rm *.line

popd
