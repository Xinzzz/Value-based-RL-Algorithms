import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))
from common.arguments import get_args

config = get_args()

print(config.gamma)