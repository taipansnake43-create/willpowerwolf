#!/usr/bin/env python3
"""Entry point for Railway deployment.

Executes the main crypto engine script.
"""
import runpy
import os

# Run the main engine script
script_path = os.path.join(os.path.dirname(__file__), "btc_engine_v22_fixed (3).py")
runpy.run_path(script_path, run_name="__main__")
