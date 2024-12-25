1. 测试环境：测试环境是三个项目共用的环境
    python:3.6.5
    cudatoolkit: 10（9应该也可以，测试的时候使用的版本是10）
    ubuntu/windows
    主要依赖及可行版本：
    pytorch 1.2.0 (1.0.1以上应该也可以，测试的时候使用的版本是1.2.0)
    pydicom 1.2.2
    simpleITK 1.2.4
    lifeline 0.24.12
    numpy 1.18.1
    opencv-python 4.2

2. 测试数据
    该项目的输入数据是detectorhit项目的部分结果输出，所以需要在运行此项目之前先运行detectorhit项目。

3. 测试步骤
3.1 手动修改配置文件
    tracer/config_tracer.py: 需要手动修改detector_path，tracer_path分别为detectorhit项目的绝对路径和tracer项目的绝对路径。
3.2 运行
    运行tracer/main_local.py。（需要首先运行detecotrhit项目。）
3.3 结果输出
    输出结果在cover_2d_out中，对于每一个输入图片xx_orgin.jpg，都会有一个对应的xx_cover.jpg以及xx_overlay.jpg。xx_cover.jpg是预测的输入图片中导致该节点为恶性节点的像素点，xx_overlay.jpg为xx_cover与xx_origin的合并图像。