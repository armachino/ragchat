import sys

def get_entry_script_name():
    try:
        return sys.modules["__main__"].__file__
    except AttributeError:
        return None
