from dotenv import load_dotenv

from .loop import main as run_loop


def main() -> None:
	"""Start the interactive terminal demo."""
	load_dotenv()
	run_loop()


if __name__ == "__main__":
	main()
