import sys

REQUIRED_PYTHON = "python3"


def main():
    """
    Check the version of python that is currently running
    and compare it with the version specified in REQUIRED_PYTHON.
    If they match, print a message indicating
    that the development environment passes all tests,
    otherwise raises a TypeError.
    """

    # Get the major version number of the system's Python interpreter
    system_major = sys.version_info.major
    # Check if REQUIRED_PYTHON is 'python' or 'python3'
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))
    # Compare system's python major version with the required version
    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version
            )
        )
    else:
        print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
