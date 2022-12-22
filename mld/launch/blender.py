# Fix blender path
import sys
import os
# local packages
sys.path.append(os.path.expanduser("~/.local/lib/python3.9/site-packages"))
import bpy
import os
from argparse import ArgumentParser

# Monkey patch argparse such that
# blender / python / hydra parsing works
def parse_args(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)

setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_args)
