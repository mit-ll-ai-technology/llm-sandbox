"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

""" Read and write functions.
"""

import pickle


def read_text_file(file_path: str):
    """Read a file to string."""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            line_count = file_content.count("\n") + 1
            return file_content, line_count
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except IOError:
        print("Error: IO error.")
        return False


def write_string_to_file(file_path: str, content: str, overwrite: bool = True):
    """Write a string to file. Used for recording the working memory."""

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    try:
        with open(file_path, mode, encoding="utf-8") as file:
            file.write(content)
        return True
    except IOError:
        print("Error: IO error.")
        return False


def dump_session_variables(filename, variables: list, variables_names: list):
    """Dump session variables to a file."""

    if variables_names == []:
        variables_names = dir()

    session_variables = {}
    # Get all variables in the current session
    for var, var_name in zip(variables, variables_names):
        print(f"saving {var_name}")
        session_variables[var_name] = var

    # Dump variables to pickle file
    with open(filename, "wb") as file:
        pickle.dump(session_variables, file)
