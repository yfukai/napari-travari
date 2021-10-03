"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Napari Travari."""


if __name__ == "__main__":
    main(prog_name="napari-travari")  # pragma: no cover
