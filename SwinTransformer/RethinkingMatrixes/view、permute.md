# pytorch维度变换(view、permute、transpose的本质)

## 1.数组保存的方式
无论是Pytorch还是Numpy，数组都是以连续的方式进行保存的。

考虑下面这个a：
```python
a = np.arange(1,17)

out:
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]

```
对a进行reshape操作后， a其实仍是以连续的方式保存
```python
a = np.reshape(4,4)

out:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
```
**那么，什么改变了呢？其实就是索引的方式**

当a的shape为(1, 16)时，a的索引只有一个，a[i],这里i的范围为0~15（a[0]~a[15])

而当a的shape变为(4, 4)的时候， a的索引变为了2个，a[i, j]， 这里i的范围是0~4，j也是0~4

**那么，当a的shape变为(2, 2, 4)的时候是怎么切分的？**
这时索引应该为a[i, j, k]

对于(2, 2, 4)里的第一个维度， 程序会把连续的数组切为2份，一份为[1  2  3  4  5  6  7  8], 另一份为[9 10 11 12 13 14 15 16]

即对于i来说，a[0,:,:]对应的**数字**应该为[1  2  3  4  5  6  7  8]，a[1,:,:]对应的**数字**应该为[9 10 11 12 13 14 15 16]

同样的，对于j来说，即是把i控制的两份又各自分为两份，分别为[1  2  3  4]、[5  6  7  8]；[9 10 11 12]、[13 14 15 16]

对k来说同样如此。

画成图的话为图一：
![image](https://user-images.githubusercontent.com/42695873/141111573-3d66575a-ae46-4260-9604-c9291cf5e6b3.png)
<p align="center">图一 矩阵每个维度的表示</p>


图一为例，索引a[0,1,1]其实就是图二
![image](https://user-images.githubusercontent.com/42695873/141111525-1d1404f9-9d79-4bd0-a035-c0570b82fdc6.png)
<p align="center">图二 具体例子的索引</p>


那么，可以看到，三个维度所用的跨度(stride)为:

<img src="https://latex.codecogs.com/svg.image?&space;stride_{i}=8" title=" stride_{i}=8" />(将16个元素分为2份，每份**8**个）

<img src="https://latex.codecogs.com/svg.image?&space;stride_{j}=4" title=" stride_{j}=4" />(将8个元素分为2份，每份**4**个）

<img src="https://latex.codecogs.com/svg.image?&space;stride_{k}=1" title=" stride_{k}=1" />(将4个元素分为4份，每份**1**个）

## 2.Pytorch和Numpy中更改矩阵维度的本质
利用view、permute更改矩阵都是将stride进行修改，即每个维度到底分为几份（几份就是几个stride）

举例，继续考虑a矩阵，将a矩阵更改为tensor，并利用view函数更改矩阵为(2,2,2,2)
那么更改后的样子应该长啥样？按照图一的表示方法，不难画出图三
![image](https://user-images.githubusercontent.com/42695873/141112174-e41e5cbe-7c68-4ef3-9c9e-890ab9d6140c.png)
<p align="center">图三 具体例子的索引</p>

那么问题来了，如果我们想让矩阵长如下的样子，应该怎么办呢？
```python
tensor([[[[ 1,  2],
          [ 5,  6]],

         [[ 3,  4],
          [ 7,  8]]],


        [[[ 9, 10],
          [13, 14]],

         [[11, 12],
          [15, 16]]]], dtype=torch.int32)
```
不如画个图，看看到底怎么变化了

