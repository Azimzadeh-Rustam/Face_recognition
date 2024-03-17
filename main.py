import os
from dotenv import load_dotenv, find_dotenv
from god_eye import GodEye


def main():
    load_dotenv(find_dotenv())
    LOGIN = os.getenv('LOGIN')
    PASSWORD = os.getenv('PASSWORD')

    God_eye = GodEye()


if __name__ == '__main__':
    main()
