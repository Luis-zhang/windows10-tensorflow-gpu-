# windows10-tensorflow-gpu-
windows10 anaconda tensorflow-gpu2.4.0安装
<div id ="0.0">目录<div>    

[tensorflow对应版本cuda和cudnn版本对照参考](#1.0)  
[anaconda安装](#1.1)
[设定anaconda镜像源](#1.2)
[安装CUDA](#1.3)
[安装cudnn](#1.4)
[配置环境变量](#1.5)
[tensorflow-gpu安装](#1.6)
[pycharm安装](#1.7)
[附上MNIST代码](#1.8)
[conda 常用命令](#2.1)
<div id = "1.0">首先确认自己需要安装哪个版本的tensorflow再安装对应版本CUDA及cudnn！！！！！！！！<div>  
本教程实例为：  
  
| tensorflow | CUDA | cuDNN | Python| Anaconda | 
| :---: | :---: | :---: | :---|  :---: |
|2.4.0 |11.0| 8.0| Python3.8 |Anaconda 2020.11|
```baidu
需要的可以自取
链接：https://pan.baidu.com/s/1UcvTWXQSgId4p9mEpccrkg 
提取码：dhcc 
```



![tf_cuda_cudnn](https://img-blog.csdnimg.cn/20210226163833933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)

[返回目录](#0.0)
## <div id = "1.1">anaconda安装<div>
从官网中下载`windows`版本[anaconda][anaconda_download]。

下载个人版本够用了，下面放的截止至文档编写完成当日最新版的`anaconda`下载链接
* `anaconda`[windows64位][windows64]下载链接
* `anaconda`[Linux64位][Linux64]下载链接  

***
> **点击`next`**
![anaconda1](https://img-blog.csdnimg.cn/20210302092937675.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  

> **选择 All user**  
>>![anaconda2](https://img-blog.csdnimg.cn/20210302093024148.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  

> **安装路径选择**`D:\ProgramFIles\anaconda`
>>![anaconda3](https://img-blog.csdnimg.cn/20210302093344950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  

>  **全部勾选** 
>>第一个将`Anaconda`添加至系统环境变量
>>将`Anaconda3`中的`Python3.8`设定为系统默认Python环境
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210302093602159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  


##  <div id = "1.2">设定anaconda镜像源<div>  
[返回目录](#0.0)
> 打开`Anaconda prompt`
> 首先创建`.condarc`文件
> `.condarc`文件默认是不存在的，但当用户第一次运行 `conda config`命令时，将会在windows系统中的`C:\Users\dhcc`路径下创建该文件。
> 用记事本打开该文件将下面任一脚本代码替代原有代码复制并保存。
>  ```conda
>  #beiji
>  channels:
>  - defaults
>show_channel_urls: true
>channel_alias: https://mirrors.bfsu.edu.cn/anaconda
>default_channels:
  >- https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  >- https://mirrors.bfsu.edu.cn/anaconda/pkgs/free
  >- https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  >- https://mirrors.bfsu.edu.cn/anaconda/pkgs/pro
  >- https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
>custom_channels:
  >conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
  >```
  >
  >```conda
  >channels:
>  - defaults
>show_channel_urls: true
>default_channels:
 > - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  >- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  >- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
>custom_channels:
> conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 >bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 >menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 >pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 >simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 >```
 >```conda  
 >#在conda prompt输入使config生效
 >conda config --set show_channel_urls yes
 >```
## <div id = "1.3">  安装CUDA<div>
[返回目录](#0.0)
[CUDA官网下载][cuda下载地址]  

> 安装路径选择``

>![cuda1](https://img-blog.csdnimg.cn/20210302110057697.png#pic_center)  
> **同意并继续**
>![cuda2](https://img-blog.csdnimg.cn/2021030211132340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
>  **选择自定义安装**
>![cuda3](https://img-blog.csdnimg.cn/20210302111426528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
>  **去掉`Visual Studio Integration`**  
>  
> 若`Driver components`中的`Display Driver`当前版本大于新版本也将钩去掉
> ![cuda4](https://img-blog.csdnimg.cn/20210302111548221.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
> `选择安装位置`选择默认即可，  记住安装路径。配置环境变量需要用到。
> ![cuda5](https://img-blog.csdnimg.cn/20210302112331398.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
> **完成安装**![cuda6](https://img-blog.csdnimg.cn/20210302112544267.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
>   
## <div id = "1.4">  安装cudnn<div>
[返回目录](#0.0)
这里放一个[cudnn官网链接][cudnn下载地址],下载`cudnn`需要注册`Nvidia`账号，可能需要科学上网。
>  解压`cudnn`并将`cuda`文件夹名改为`cudnn`放在`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0`目录下
> ![cudnn1](https://img-blog.csdnimg.cn/20210302114706275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
 ## <div id = "1.5">  配置环境变量<div>
 >将`CUDA`和`cuDNN`库添加到环境变量中
 >```path
 >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\cudnn\bin
 >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64
 >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin
 >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\libnvvp
 >```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210302143346343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)
> 配置环境变量后**重启电脑**     


## <div id = "1.6">tensorflow-gpu安装<div>
[返回目录](#0.0)
> 1. 打开`Anaconda Navigator`
>2. 点击<kbd>Environments</kbd>
>![environment1](https://img-blog.csdnimg.cn/20210227115330832.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
>点击`creat`
>![create1](https://img-blog.csdnimg.cn/20210227120458955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
>4.输入环境变量名，选择`python`版本，点击<kbd>create</kbd>。稍等一会便创建成功  
>![env2](https://img-blog.csdnimg.cn/20210227120938812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)
>打开`conda prompt`
>输入`conda update conda`
>然后输入`pip install tensorflow-gpu==2.4.0`安装`tensorflow2.4.0`
### 验证tensorflow-gpu是否安装成功
```Python3
(tensorflow1) C:\Users\dhcc>python
Python 3.8.6 | packaged by conda-forge | (default, Jan 25 2021, 22:54:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2021-02-27 16:24:32.243502: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
>>> tf.test.is_gpu_available()
2021-02-27 16:27:25.706998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:83:00.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.88GiB deviceMemoryBandwidth: 836.37GiB/s
2021-02-27 16:27:25.719334: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2021-02-27 16:27:25.725227: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-02-27 16:27:25.731455: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-02-27 16:27:25.737729: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2021-02-27 16:27:25.744441: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2021-02-27 16:27:25.752021: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2021-02-27 16:27:25.759194: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2021-02-27 16:27:25.766735: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-02-27 16:27:25.782919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2021-02-27 16:27:26.528352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-02-27 16:27:26.534888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0
2021-02-27 16:27:26.539445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N
2021-02-27 16:27:26.558320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/device:GPU:0 with 14827 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:83:00.0, compute capability: 7.0)
2021-02-27 16:27:26.569970: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
True
```




#### 或者输入
```Python
>>> tf.config.list_physical_devices('GPU')
2021-02-27 16:40:04.176880: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-27 16:40:04.187789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:83:00.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.88GiB deviceMemoryBandwidth: 836.37GiB/s
2021-02-27 16:40:04.199739: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2021-02-27 16:40:04.205925: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-02-27 16:40:04.211750: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2021-02-27 16:40:04.218167: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2021-02-27 16:40:04.224249: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2021-02-27 16:40:04.229874: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2021-02-27 16:40:04.235953: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2021-02-27 16:40:04.242297: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2021-02-27 16:40:04.259506: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
### <div id = "1.7">安装Pycharm<div>
[返回目录](#0.0)
安装Pycharm一直点下一步就可以
安装完成后需要将`anaconda`环境变量添加进pycharm中
1. 打开pycharm软件，点击`Create New Project`选项
2. 在弹出的界面中点击右侧的`Interpreter`下拉框后面的按钮
3. 进入解释器设置界面，我们选择`Conda Environment`后面的按钮
4. 选择`Exiting environment`
5. 选择anaconda所需要的环境中的python.exe
6. ![pycharm1](https://img-blog.csdnimg.cn/20210302150705689.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70#pic_center)  
7. 点击OK完成配置



#### <div id = "1.8">附上MNIST代码<div>  
[返回目录](#0.0)


```
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets



(x, y), (x_val, y_val) = datasets.mnist.load_data() 
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

 


model = keras.Sequential([ 
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.Adam(learning_rate=0.001)


def train_epoch(epoch):

    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):


        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())



def train():

    for epoch in range(30):

        train_epoch(epoch)






if __name__ == '__main__':
    train()

```
 
 ### 运行过程展示
 ![MNIST1](https://img-blog.csdnimg.cn/20210302151254421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1aXNfamll,size_16,color_FFFFFF,t_70)
### <div id = "2.1">conda 常用命令<div>
[返回目录](#0.0)
* Conda是没有重命名环境的功能的, 要实现这个基本需求, 只能通过克隆-删除的过程。
* 切记不要直接mv移动环境的文件夹来重命名, 会导致一系列无法想象的错误的发生!
```conda
1. conda --version #查看conda版本，验证是否安装

2. conda update conda #更新至最新版本，也会更新其它相关包

3. conda update --all #更新所有包

4. conda update package_name #更新指定的包

5. conda create -n env_name package_name #创建名为env_name的新环境，并在该环境下安装名为package_name 的包
可以指定新环境的版本号，例如：
conda create -n python2 python=python2.7 numpy pandas，
创建了python2环境，python版本为2.7，同时还安装了numpy pandas包

6. source activate env_name #切换至env_name环境

7. source deactivate #退出环境

8. conda info -e #显示所有已经创建的环境

9. conda create --name new_env_name --clone old_env_name #复制old_env_name为new_env_name

10. conda remove --name env_name –all #删除环境

11. conda list #查看所有已经安装的包

12. conda install package_name #在当前环境中安装包

13. conda install --name env_name package_name #在指定环境中安装包

14. conda remove -- name env_name package #删除指定环境中的包

15. conda remove package #删除当前环境中的包

16. conda create -n tensorflow_env tensorflow

conda activate tensorflow_env #conda 安装tensorflow的CPU版本

17. conda create -n tensorflow_gpuenv tensorflow-gpu

conda activate tensorflow_gpuenv #conda安装tensorflow的GPU版本

18. conda env remove -n env_name #采用第10条的方法删除环境失败时，可采用这种方法

19. conda install --channel https://conda.anaconda.org/anaconda tensorflow=1.8.0 
提供一个下载地址，使用上面命令就可安装1.8.0版本tensorflow

conda clean -p 删除一些没用的包，这个命令会检查哪些包没有在包缓存中被硬依赖到其他地方，并删除它们

conda clean -t可以将conda保存下来的tar打包。

conda clean -y -all //删除所有的安装包及cache
```

 


 
 





 







 





















































[anaconda_download]: https://www.anaconda.com/products/individual
[windows64]: https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe
[Linux64]:https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
[cuda下载地址]: https://developer.nvidia.com/zh-cn/cuda-downloads
[cuda_11.2.1_461.09_win10.exe]: https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_461.09_win10.exe
[cudnn下载地址]:https://developer.nvidia.com/rdp/cudnn-download
