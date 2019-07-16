# CS344 LAB

### 环境配置

#### 1. 安装CUDA和NVIDIA驱动

从官网下载CUDA的安装包 如：`cuda_9.0.176_384.81_linux.run`

执行命令 `./cuda_9.0.176_384.81_linux.run`

建议先不选择安装NVIDIA驱动的选项 安装好了CUDA之后再安装

[参考链接]: https://blog.csdn.net/wanzhen4330/article/details/81699769

接下来安装NVIDIA驱动：

用`sudo  ubuntu-drivers devices`命令来查看自己的显卡适合的驱动型号，然后去英伟达官网（https://www.nvidia.cn/Download/index.aspx?lang=cn）下载相应驱动的安装包，如

`NVIDIA-Linux-x86_64-430.26.run`，然后执行`./NVIDIA-Linux-x86_64-430.26.run`运行安装程序（注意驱动安装可能要求特定版本的gcc，可能和cuda编译需要的gcc版本不一致，所以可以先安装相应版本的gcc，安装完驱动后再换回来）

#### 2. 配置代码运行的c++编译环境

首先需要基本的gcc 和 g++环境。

有些lab需要opencv 库的支持，opencv的安装和配置参考：

[openCV 安装和配置]: https://blog.csdn.net/Wangguang_/article/details/85762705

安装好了opencv之后，要想正确运行make 来编译自己的程序，还需要修改一下Makefile

![](/home/volcano/Pictures/makefile.png)

把里面像`CUDA_INCLUDEPATH=/usr/local/cuda-9.0/include`这样的地址改成自己相应库的地址（如果是默认路径安装一般只需要改一下cuda的版本号就好）



要想添加新的opencv库 在

`OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_imgcodecs -lopencv_videoio`后面添加`-l`+相应的库的名字

除此之外，可能因为显卡和CUDA的版本不同，`NVCC_OPTS=-O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -m64`中的 -arch 的参数也不一样，它默认是sm_20, 我运行时就需要 sm_60