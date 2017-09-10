# -*- coding: utf-8 -*-
import random
import time
import numpy as np
import graph_cut

INF = 1e30

class GM():
    def __init__(self, unary, hor_c, vert_c, metric):
        """
        Конструктор, принимает на вход параметры энергии модели.

        Вход:
        unary -- массив N x M x K вещественных чисел, унарные потенциалы
        hor_c -- массив N x (M - 1) вещественных чисел, параметры парных потенциалов горизонтальных рёбер графа
        vert_c -- массив (N - 1) x M вещественных чисел, параметры парных потенциалов вертикальных рёбер графа
        metric -- массив K x K, метрика на K точках. Используется при определении парных потенциалов
        """
        self.unary = unary
        self.hor_c = hor_c
        self.vert_c = vert_c
        self.metric = metric

    def energy(self, labels):
        """
        Для данного набора значений переменных графической модели вычисляет энергию.

        Вход:
        labels -- массив N x M целых чисел от 0 до K-1, значения переменных графической модели
        Выход:
        energy -- вещественное число, энергия на данном наборе переменных
        """
        ind = np.indices(labels.shape)
        unary_part = np.sum(self.unary[ind[0], ind[1], labels])
        binary_hor_part = np.sum(self.hor_c * self.metric[labels[:,:-1], labels[:,1:]])
        binary_vert_part = np.sum(self.vert_c * self.metric[labels[:-1,:], labels[1:,:]])
        return unary_part + binary_hor_part + binary_vert_part

    def expand_alpha(self, labels, alpha):
        """
        Вычисляет один шаг алгоритма alpha_expansion. Для данного набора значений переменных labels минимизирует энергию, расширяя множество принимающих значение alpha переменных.

        Вход:
        labels -- массив N x M целых чисел от 0 до K - 1, значения переменных графической модели
        alpha -- целое число от 0 до K - 1, параметр расширения
        Выход:
        new_labels -- массив N x M целых чисел от 0 до K - 1, новый набор значений переменных
        """
        flow, result = graph_cut.graph_cut(*self._build_graph(labels, alpha))
        new_labels = labels.copy()
        new_labels[result.reshape(labels.shape) == 0] = alpha
        return new_labels

    def _build_graph(self, labels, alpha):
        """
        Вспомогательная процедура для метода expand_alpha. Для данного набора значений переменных labels и значения alpha сводит задачу минимизации энергии к поиску минимального разреза в графе. 

        Вход:
        labels -- массив N x M целых чисел от 0 до K - 1, значения переменных графической модели
        alpha -- целое число от 0 до K - 1, параметр расширения
        Выход:
        term_weights -- массив N x M x 2 вещественных чисел, пропускные способности терминальных рёбер графа
        edge_weights -- массив (N - 1) * M + N * (M - 1) x 4 вещественных чисел, пропускные способности рёбер, соединяющих нетерминальные вершины
        """
        ind = np.indices(labels.shape)
        
        term_weights = np.zeros((labels.shape[0], labels.shape[1], 2))
        
        source = self.unary[ind[0], ind[1], labels]
        source[labels == alpha] = INF
        
        term_weights[:, :, 0] = source
        term_weights[:, :, 1] = self.unary[:, :, alpha]
        
        m00_hor = self.hor_c * self.metric[labels[:,:-1], labels[:,1:]]
        m01_hor = self.hor_c * self.metric[labels[:,:-1], alpha]
        m10_hor = self.hor_c * self.metric[alpha, labels[:,1:]]
        m11_hor = self.hor_c * self.metric[alpha, alpha]

        a_hor = (m01_hor - m10_hor + m00_hor + m11_hor) / 4.0
        b_hor = (m10_hor - m01_hor + m00_hor + m11_hor) / 4.0
        e_hor = (m01_hor + m10_hor - m00_hor - m11_hor) / 2.0

        term_weights[ind[0][:,:-1], ind[1][:,:-1], 0] += a_hor
        term_weights[ind[0][:,1:], ind[1][:,1:], 0] += b_hor
        term_weights[ind[0][:,:-1], ind[1][:,:-1], 1] += b_hor
        term_weights[ind[0][:,1:], ind[1][:,1:], 1] += a_hor

        m00_vert = self.vert_c * self.metric[labels[:-1,:], labels[1:,:]]
        m01_vert = self.vert_c * self.metric[labels[:-1,:], alpha]
        m10_vert = self.vert_c * self.metric[alpha, labels[1:,:]]
        m11_vert = self.vert_c * self.metric[alpha, alpha]

        a_vert = (m01_vert - m10_vert + m00_vert + m11_vert) / 4.0
        b_vert = (m10_vert - m01_vert + m00_vert + m11_vert) / 4.0
        e_vert = (m01_vert + m10_vert - m00_vert - m11_vert) / 2.0

        term_weights[ind[0][:-1,:], ind[1][:-1,:], 0] += a_vert
        term_weights[ind[0][1:,:], ind[1][1:,:], 0] += b_vert
        term_weights[ind[0][:-1,:], ind[1][:-1,:], 1] += b_vert
        term_weights[ind[0][1:,:], ind[1][1:,:], 1] += a_vert

        ind = np.arange(labels.shape[0] * labels.shape[1]).reshape(labels.shape)
        edge_weights_hor = np.vstack([ind[:,:-1].ravel(), ind[:,1:].ravel(), e_hor.ravel(), e_hor.ravel()]).T
        edge_weights_vert = np.vstack([ind[:-1,:].ravel(), ind[1:,:].ravel(), e_vert.ravel(), e_vert.ravel()]).T

        return term_weights.reshape(labels.shape[0] * labels.shape[1], 2), np.vstack([edge_weights_hor, edge_weights_vert])


def alpha_expansion(gm, labels, max_iter=50, display=False, rand_order=True):
    """
    Алгоритм alpha_expansion.

    Вход:
    gm -- объект класса GM
    labels -- массив N x M целых чисел от 0 до K - 1, начальные значения переменных
    max_iter -- целое число, максимальное число итераций алгоритма
    display -- булева переменная, алгоритм выводит вспомогательную информацию при display=True
    rand_order -- булева переменная, при rand_order=True на каждой итерации alpha берется в случайном порядке
    Выход:
    new_labels -- массив N x M целых чисел от 0 до K - 1, значения переменных, на которых завершил работу алгоритм
    energies -- массив max_iter вещественных чисел, значения энергии перед каждой итерацией алгоритма
    times -- массив max_iter вещественных чисел, время вычисления каждой итерации алгоритма
    """
    K = gm.unary.shape[2]
    alpha = K - 1
    energies = []
    times = []
    if display:
        print "Initial energy: {}.".format(gm.energy(labels))
    for i in xrange(max_iter):
        if rand_order:
            alpha = random.randint(0, K - 1)
        else:
            alpha = (alpha + 1) % K
        timestamp = time.time()
        new_labels = gm.expand_alpha(labels, alpha)
        times.append(time.time() - timestamp)
        labels = new_labels
        energies.append(gm.energy(labels))
        if display:
            print "Iteration {}. Time: {:0.4f} s. Energy: {:0.4f}.".format(i + 1, times[-1], energies[-1])
    return labels, energies, times


def define_energy(images, seeds):
    """
    Вычисляет параметры графической модели для K изображений images разрешения N x M и набора семян seeds.
    
    Вход:
    images -- массив N x M x C x K вещественных чисел, C - число каналов.
    seeds -- массив N x M x K булевых переменных, seeds[n, m, k] = True должно поощрять выбор k изображения на позиции n x m при склеивании изображений.
    Выход:
    unary -- массив N x M x K вещественных чисел, унарные потенциалы
    hor_c -- массив N x (M - 1) вещественных чисел, параметры парных потенциалов горизонтальных рёбер графа
    vert_c -- массив (N - 1) x M вещественных чисел, параметры парных потенциалов вертикальных рёбер графа
    metric -- массив K x K, метрика на K точках. Используется при определении парных потенциалов
    """
    N, M, C, K = images.shape
    unary = 1000 * K * (K + 1) * np.ones(seeds.shape)
    unary[seeds] = 0
    hor_c = np.zeros((N, M - 1))
    vert_c = np.zeros((N - 1, M))
    for i in xrange(K):
        for j in xrange(i + 1, K):
            hor_c += np.sum((images[:, 1:, :, i] - images[:, :-1, :, j]) ** 2, axis=2) ** 2
            vert_c += np.sum((images[1:, :, :, i] - images[:-1, :, :, j]) ** 2, axis=2) ** 2
    metric = (np.ones((K, K)) - np.eye(K))
    return unary, hor_c, vert_c, metric

def stitch_images(images, seeds):
    """
    Процедура для склеивания изображений.

    Вход:
    images -- массив N x M x C x K вещественных чисел, C - число каналов.
    seeds -- массив N x M x K булевых переменных, seeds[n, m, k] = True должно поощрять выбор k изображения на позиции n x m при склеивании изображений.
    Выход:
    res -- массив N x M x C вещественных чисел, склеенное изображение
    mask -- массив N x M целых чисел от 0 до K - 1, mask[n, m, k] равен номеру изображения, из которого взят пиксель на позиции n x m.
    """
    labels = np.random.choice(images.shape[3], size=(images.shape[0], images.shape[1]))
    ind = np.indices(labels.shape)
    labels, _, _ = alpha_expansion(GM(*define_energy(images, seeds)), labels, max_iter=12, display=True, rand_order=False)
    return images[ind[0], ind[1], :, labels], labels

def gauss_filter(base, images, labels, type, filter_size, sigma):
    INF = 1e6
    result = np.zeros(images.shape[:3])
    filter = np.arange(filter_size * 2 + 1) - filter_size
    weights = np.exp(- (sigma / float(filter_size)) * np.abs(filter))
    for i in xrange(result.shape[0]):
        for j in xrange(result.shape[1]):
            cur_label = labels[i, j]
            window = filter.copy()
            if type == 0:
                window += i
                window[window < 0] = 0
                window[window >= result.shape[0]] = result.shape[0] - 1
                window = labels[window, j]
            else:
                window += j
                window[window < 0] = 0
                window[window >= result.shape[1]] = result.shape[1] - 1
                window = labels[i, window]
            value = 0.0
            norm = 1.0
            for k in xrange(images.shape[3]):
                if k == cur_label:
                    value += base[i, j, :]
                    continue
                cur_w = weights[window == k]
                if len(cur_w) > 0:
                    cur_w = cur_w.max()
                    value += cur_w * images[i, j, :, k]
                    norm += cur_w     
            result[i, j, :] = value / norm
    return result

def filter_stitch(base, images, labels, filter_size=16, sigma=4.0):
    tmp = gauss_filter(base, images, labels, 0, filter_size=filter_size, sigma=sigma)
    return gauss_filter(tmp, images, labels, 1, filter_size=filter_size, sigma=sigma)
