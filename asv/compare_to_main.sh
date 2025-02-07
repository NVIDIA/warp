#!/bin/bash
set -e

asv continuous --append-samples --interleave-rounds --no-only-changed main $(git rev-parse HEAD 2>/dev/null)
