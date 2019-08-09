# 做兼容的
from __future__ import print_function
import numpy as np
# 处理图片的
from PIL import Image
# 系统变量，接收命令行传递的值
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
一个使用概率图模型对图片进行降噪的
"""


def compute_log_prob_helper(Y, i, j):
    """
    查看Y的下表是否过界，过了返回0
    :param Y:
    :param i:
    :param j:
    :return:
    """
    try:
        return Y[i][j]
    except IndexError:
        return 0


def compute_log_prob(X, Y, i, j, w_e, w_s, y_val):
    """

    :param X: 观察值
    :param Y: 为观察值
    :param i:
    :param j:
    :param w_e: 大于零的参数
    :param w_s: 大于零的参数
    :param y_val: -1 1 ？？？
    :return:
    """
    # 这是验证观测值和未观察的对比，如果相同则为正
    result = w_e * X[i][j] * y_val
    # 这是计算四个方向，计算误差？顶点对于邻近的节点具有相似性，这也算是用到了马尔可夫随机场
    result += w_s * y_val * compute_log_prob_helper(Y, i - 1, j)
    result += w_s * y_val * compute_log_prob_helper(Y, i + 1, j)
    result += w_s * y_val * compute_log_prob_helper(Y, i, j - 1)
    result += w_s * y_val * compute_log_prob_helper(Y, i, j + 1)
    # todo:做什么用的？？？
    return result


def denoise_image(X, w_e, w_s):
    """
    为图片矩阵降噪
    :param X:
    :param w_e:
    :param w_s:
    :return:
    """
    m, n = np.shape(X)
    # 初始化未观察参数
    Y = np.copy(X)
    max_iter = 10 * m * n
    for iter in range(max_iter):
        # 随机选择位置
        i = np.random.randint(m)
        j = np.random.randint(n)
        # 计算两个Y_ij值的对数概率，在未知真实图的数值下分别对正反例的概率进行计算。
        log_p_neg = compute_log_prob(X, Y, i, j, w_e, w_s, -1)
        log_p_pos = compute_log_prob(X, Y, i, j, w_e, w_s, 1)

        if log_p_neg > log_p_pos:
            Y[i][j] = -1
        else:
            Y[i][j] = 1
        if iter % 100000 == 0:
            print("进行中")
    return Y


def read_image_and_binarize(image_file):
    im = Image.open(image_file).convert("L")
    A = np.asarray(im).astype(int)
    A.flags.writeable = True
    # todo:128指什么？？？
    A[A < 128] = -1
    A[A >= 128] = 1
    return A


def add_noise(orig):
    A = np.copy(orig)
    for i in range(np.size(A, 0)):
        for j in range(np.size(A, 1)):
            r = np.random.rand()
            if r < 0.1:
                A[i][j] = -A[i][j]
    return A


def convert_from_matrix_and_save(M, filename, display=False):
    M[M == -1] = 0
    M[M == 1] = 255
    im = Image.fromarray(np.uint8(M))
    if display:
        im.show()
    im.save(filename)


def get_mismatched_percentage(orig_image, denoised_image):
    diff = abs(orig_image - denoised_image) / 2
    return (100.0 * np.sum(diff)) / np.size(orig_image)


def main():
    orig_image = read_image_and_binarize("input.png")
    # if len(sys.argv) > 2:
    #     try:
    #         w_e = eval(sys.argv[2])
    #         w_s = eval(sys.argv[3])
    #     except:
    #         print("except")
    #         sys.exit()
    # else:
    w_e = 8
    w_s = 10

    noisy_image = add_noise(orig_image)

    denoised_image = denoise_image(noisy_image, w_e, w_s)
    print(get_mismatched_percentage(orig_image, denoised_image))
    convert_from_matrix_and_save(orig_image, "orig_image.png", display=False)
    convert_from_matrix_and_save(noisy_image, "noisy_image.png", False)
    convert_from_matrix_and_save(denoised_image, "denoise_image.png", False)


if __name__ == '__main__':
    main()
