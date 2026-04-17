#!/bin/sh
# Fetch a sample AMI meeting recording (ES2016a) and the corpus license.
set -eu

mkdir -p amicorpus/ES2016a/audio

wget -P amicorpus/ES2016a/audio \
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2016a/audio/ES2016a.Mix-Headset.wav

wget -O amicorpus/CCBY4.0.txt \
  https://groups.inf.ed.ac.uk/ami/download/temp/../CCBY4.0.txt
