"""Run the released Pearl / DENEB command line without its dead download.

Both cli.py files call model2download() while the module is imported, to build
the choices of an unrelated `download` command.  That fetches
https://unbabel-experimental-models.s3.amazonaws.com/comet/share/public-models.yaml,
which now answers 403, so the released `cli.py train` cannot start.  This
replaces that one lookup with an empty model list and then hands over to the
released command group unchanged.

    released_cli.py <pearl|deneb> train -f configs/config.yaml
"""
import importlib
import sys


def main():
    package = sys.argv[1]
    models = importlib.import_module(f'{package}.models')
    models.model2download = lambda *args, **kwargs: {}
    cli = importlib.import_module(f'{package}.cli')
    sys.argv = [package] + sys.argv[2:]
    getattr(cli, package)()


if __name__ == '__main__':
    main()
