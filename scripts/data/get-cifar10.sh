#!/bin/sh
# Downloads the binary version of CIFAR-10 into data/cifar10/, checks the archive against the
# MD5 published on https://www.cs.toronto.edu/~kriz/cifar.html, and unpacks its six batch
# files for `cavitation_datasets::cifar10::load`.
# Usage: scripts/data/get-cifar10.sh   (run from anywhere; needs curl or wget, md5sum or
# macOS's md5, and tar)
set -eu

url="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
md5="c32a1d4ab5d03f1284b67883e8d87530"
dir="$(cd "$(dirname "$0")/../.." && pwd)/data/cifar10"
archive="$dir/cifar-10-binary.tar.gz"
mkdir -p "$dir"

# Prints the MD5 of file $1: md5sum on Linux, md5 on macOS and the BSDs.
md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

# Downloads URL $1 to file $2 with curl, or wget if there is no curl.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$2" "$1"
    else
        wget -q -O "$2" "$1"
    fi
}

missing=""
for name in data_batch_1.bin data_batch_2.bin data_batch_3.bin data_batch_4.bin \
    data_batch_5.bin test_batch.bin; do
    [ -f "$dir/$name" ] || missing="$missing $name"
done
if [ -z "$missing" ]; then
    echo "CIFAR-10: already there"
    exit 0
fi

if [ ! -f "$archive" ] || [ "$(md5_of "$archive")" != "$md5" ]; then
    echo "cifar-10-binary.tar.gz: downloading (162 MB)"
    fetch "$url" "$archive"
fi
actual="$(md5_of "$archive")"
if [ "$actual" != "$md5" ]; then
    rm -f "$archive"
    echo "cifar-10-binary.tar.gz: MD5 is $actual, expected $md5" >&2
    exit 1
fi
# The archive holds cifar-10-batches-bin/; its files go straight into data/cifar10/.
tar -xzf "$archive" -C "$dir" --strip-components=1
rm -f "$archive"
echo "CIFAR-10 is in $dir"
