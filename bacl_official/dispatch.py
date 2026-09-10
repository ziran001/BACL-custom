"""Keep the two implementations separate, including their dependencies."""
import argparse
import importlib
import sys


def dispatch(command):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('official', 'torchvision'), default='official')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if args.backend == 'torchvision':
        print('Legacy TorchVision approximation; evaluation is AP50, not official LVIS AP.', flush=True)
        module = importlib.import_module('tools.{}_torchvision'.format(command))
        module.main()
    else:
        from . import cli
        getattr(cli, command)()
