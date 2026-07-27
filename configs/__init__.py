#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Package initializer for rsa.utils: import all modules in this package.

This dynamically imports all submodules in the package so they are
available as attributes of rsa.utils.
"""

from importlib import import_module
import pkgutil
from pathlib import Path

__all__ = []

package = __name__
package_path = Path(__file__).resolve().parent

for finder, name, ispkg in pkgutil.iter_modules([str(package_path)]):
	if name.startswith("__"):
		continue
	module = import_module(f".{name}", package=package)
	globals()[name] = module
	__all__.append(name)
