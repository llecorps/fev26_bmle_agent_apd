"""Validation AST du code Python généré par le LLM avant exécution.

Le code généré est un script autonome (il fait ses propres imports pandas/numpy
puis imprime un JSON). On ne peut donc pas interdire TOUS les imports comme dans
une sandbox à namespace injecté : on autorise une whitelist de modules d'analyse
de données et on rejette tout le reste (os, sys, subprocess, socket, ...).

Couches de défense (cf. explore.py) :
1. Validation AST statique (ici) : modules hors whitelist, builtins dangereux,
   dunders, méthodes d'écriture fichier/système -> rejet avant exécution.
2. subprocess `python -I` (isolé) + timeout dur côté explore.py.
3. Montage du volume data en lecture seule (docker-compose).
"""

import ast

# Modules autorisés à l'import dans le code généré (analyse de données pure).
ALLOWED_MODULES = {
    "pandas", "numpy", "json", "math", "statistics", "datetime",
    "collections", "itertools", "functools", "re", "decimal", "plotly",
}

# Builtins dont l'usage ouvre une évasion (exécution de code, accès fichier, introspection).
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint", "exit", "quit",
    "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "memoryview", "help",
}

# Méthodes/attributs qui écrivent sur disque, lancent des process ou touchent le système.
FORBIDDEN_ATTRIBUTES = {
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_sql", "to_hdf",
    "to_feather", "to_stata", "to_clipboard", "to_latex",
    "system", "popen", "remove", "rmtree", "unlink", "rename", "chmod",
    "write_image", "write_html", "write_json", "savefig",
}


class CodeValidationError(Exception):
    """Le code généré viole la politique de sécurité — rejeté avant exécution."""


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def validate_code(code: str) -> None:
    """Lève CodeValidationError si le code enfreint la politique."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"Syntaxe Python invalide : {exc}") from exc

    for node in ast.walk(tree):
        # Imports : uniquement les modules de la whitelist.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_root(alias.name) not in ALLOWED_MODULES:
                    raise CodeValidationError(f"Import interdit : {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and _module_root(node.module) not in ALLOWED_MODULES:
                raise CodeValidationError(f"Import interdit : {node.module}")

        # Boucles infinies potentielles (le timeout couvre, mais autant couper court).
        elif isinstance(node, ast.While):
            raise CodeValidationError("Boucle `while` interdite")

        # Builtins dangereux et identifiants en _.
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise CodeValidationError(f"Nom interdit : {node.id}")
            if node.id.startswith("__"):
                raise CodeValidationError(f"Identifiant dunder interdit : {node.id}")

        # Accès attribut : méthodes d'I/O et dunders.
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise CodeValidationError(f"Attribut interdit : .{node.attr}")
            if node.attr.startswith("__"):
                raise CodeValidationError(f"Attribut dunder interdit : .{node.attr}")
