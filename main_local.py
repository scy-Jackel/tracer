import base64

import os
from config_tracer import config_tracer
from main_func import view_new, view_single
from main_func import view, train

def start_train():
    train()
    print("train complete.")

def test_all():
    view()
    print("test complete.")

def test_single():
    # 本地测试文件
    patientname = "feichuang"
    view_new(patientname, patientname)


if __name__ == '__main__':
    test_all()
    # train()
    # test_single()
