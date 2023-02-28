# Sys.setenv(RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python")
library(reticulate)
py_config()

source_python("level-sets/level_sets/utils.py")
img = load_image("mnist/img_16.jpg")
