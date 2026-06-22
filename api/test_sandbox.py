"""Tests du contrat de sécurité de la validation AST (api/sandbox.py).

Lancer : pytest api/test_sandbox.py
"""

import pytest

from sandbox import CodeValidationError, validate_code

LEGITIMATE = [
    # Analyse pandas standard imprimant du JSON.
    "import pandas as pd\nimport json\n"
    "df = pd.read_parquet('/data/processed/apd_clean.parquet')\n"
    "print(json.dumps(df.groupby('Agence')['ratio_don'].mean().to_dict(), ensure_ascii=False))",
    # numpy autorisé.
    "import pandas as pd\nimport numpy as np\nimport json\n"
    "df = pd.read_parquet('/data/processed/apd_clean.parquet')\n"
    "print(json.dumps({'m': float(np.mean(df['ratio_don']))}))",
]

DANGEROUS = [
    "import os\nos.listdir('/')",
    "import sys\nsys.exit(1)",
    "import subprocess\nsubprocess.run(['ls'])",
    "import socket",
    "import requests\nrequests.get('http://evil')",
    "import shutil\nshutil.rmtree('/data')",
    "print(open('/etc/passwd').read())",
    "eval('1+1')",
    "exec('x=1')",
    "__import__('os')",
    "print((1).__class__.__mro__)",
    "getattr(__builtins__, 'eval')",
    "import pandas as pd\npd.read_parquet('x').to_csv('/tmp/exfil.csv')",
    "import pandas as pd\npd.read_parquet('x').to_pickle('/tmp/x.pkl')",
    "while True:\n    pass",
]


@pytest.mark.parametrize("code", LEGITIMATE)
def test_legitimate_code_accepted(code):
    validate_code(code)  # ne lève pas


@pytest.mark.parametrize("code", DANGEROUS)
def test_dangerous_code_rejected(code):
    with pytest.raises(CodeValidationError):
        validate_code(code)


def test_syntax_error_rejected():
    with pytest.raises(CodeValidationError):
        validate_code("print((")
