#!/bin/sh
# Downloads the four MNIST files into data/mnist/, checks each against the MD5 that
# torchvision publishes for it, and decompresses them for `cavitation_datasets::mnist::load`.
# Usage: scripts/data/get-mnist.sh   (run from anywhere; needs curl or wget, md5sum or
# macOS's md5, and gunzip)
set -eu

base_url="https://ossci-datasets.s3.amazonaws.com/mnist"
dir="$(cd "$(dirname "$0")/../.." && pwd)/data/mnist"
mkdir -p "$dir"

# "file md5-of-the-.gz" pairs, as in torchvision.datasets.MNIST.resources.
files="train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873
train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432
t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3
t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c"

# Prints the MD5 of file $1: md5sum on Linux, md5 on macOS and the BSDs.
md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$2" "$1"
    else
        wget -q -O "$2" "$1"
    fi
}

echo "$files" | while read -r name md5; do
    if [ -f "$dir/$name" ]; then
        echo "$name: already there"
        continue
    fi
    echo "$name: downloading"
    fetch "$base_url/$name.gz" "$dir/$name.gz"
    actual="$(md5_of "$dir/$name.gz")"
    if [ "$actual" != "$md5" ]; then
        rm -f "$dir/$name.gz"
        echo "$name.gz: MD5 is $actual, expected $md5" >&2
        exit 1
    fi
    gunzip -f "$dir/$name.gz"
done
echo "MNIST is in $dir"
