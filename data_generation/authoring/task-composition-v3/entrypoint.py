"""Include portable device reservations without changing pinned runner code."""
import argparse
import sys

import run
from device_batch import install_reservations, reservation_lock


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--continue-from', type=str)
    options, argv = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    install_reservations()
    with reservation_lock():
        if options.continue_from is not None:
            from codex_handoff import arguments
            try:
                argv = arguments(options.continue_from, argv)
            except (ValueError, OSError, RuntimeError) as error:
                parser.exit(1, f'Codex handoff failed: {error}\n')
        previous = sys.argv
        sys.argv = [previous[0], *argv]
        try:
            return run.main()
        except SystemExit as error:
            if error.code == 0 and any(value in ('-h', '--help') for value in argv):
                print('\nAdditional launcher option:\n'
                      '  --continue-from PATH  Hand off a saved run to a separate Codex output root.\n'
                      '                        With this option, --count is the dataset total,\n'
                      '                        including already generated source compositions.')
            raise
        finally:
            sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
