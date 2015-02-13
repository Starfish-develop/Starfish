# See whether the config loading works
import pytest

# Make sure the import triggers a userwarning
def test_config_not_found(recwarn):
    import Starfish
    w = recwarn.pop(UserWarning)
    print(w.message)
    print(w.filename)
    print(w.lineno)
