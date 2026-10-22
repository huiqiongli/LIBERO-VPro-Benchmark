#!/usr/bin/env bash
# Source this release-owned adapter setup; user runtime overrides are preserved.
_vpro_release=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_vpro_exports=$(PYTHONPATH="${_vpro_release}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m libero_vpro.runtime --shell) || return $?
eval "$_vpro_exports"
unset _vpro_exports _vpro_release
