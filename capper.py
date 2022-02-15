import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.animation as animation

R_CUT = {'min': 0.3, 'max': 2.0}
SIGMA = 1.42 / (2 ** (1 / 6))
EPSILON = 2

M = 1
DT = 0.01


def force(coord, coord_others):
    f = np.array([0.0, 0.0, 0.0])
    for coord_other in coord_others:
        r = np.linalg.norm(coord - coord_other)
        vec = coord - coord_other
        if r < R_CUT['min']:
            f += vec * 4 * EPSILON * (12 * (SIGMA ** 12 / R_CUT['min'] ** 13) - 6 * (SIGMA ** 6 / R_CUT['min'] ** 7))
        else:
            f += vec * 4 * EPSILON * (12 * (SIGMA ** 12 / r ** 13) - 6 * (SIGMA ** 6 / r ** 7))
    return f


@dataclass
class Atom:
    dic_obj = {}
    dic_coord = {}
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray
    tag: str

    def __post_init__(self):
        self.r_pre = self.r
        self.v_pre = self.v
        self.a_pre = self.a

        Atom.dic_obj[self.tag] = self

    def move(self):
        f = self.calc_force()
        if 'capper' in self.tag:  #キャップ部のみ動く
            self.a = f / M
            self.v += self.a * DT
            self.r += self.v * DT

    def update(self):
        self.r_pre = self.r
        self.v_pre = self.v
        self.a_pre = self.a

    def calc_force(self):
        near = []
        for tag, atom in Atom.dic_obj.items():
            if tag == self.tag:
                continue
            d = np.linalg.norm(atom.r_pre - self.r_pre)
            if d < R_CUT['max']:
                near.append(atom.r_pre)
        f = force(self.r_pre, near)
        return f


def set_aspect_equal_3d(ax):
    x = ax.get_xlim()
    y = ax.get_ylim()
    z = ax.get_zlim()
    mid_x = (x[0] + x[1]) / 2
    mid_y = (y[0] + y[1]) / 2
    mid_z = (z[0] + z[1]) / 2
    ran_x = abs(x[0] - x[1])
    ran_y = abs(y[0] - y[1])
    ran_z = abs(z[0] - z[1])
    max_range = max(ran_x, ran_y, ran_z) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect((1, 1, 1))


def main(n, m):
    data = pd.read_csv(f'./data/atoms_{n}_{m}.csv', index_col=0)
    for row, item in data.iterrows():
        Atom(item.values,
             np.zeros(3),
             np.zeros(3),
             row)

    d = 1.42 * np.sqrt(3 * (n ** 2 + m ** 2 + n * m)) / np.pi
    print(d)
    for y in np.linspace(-d/2, 0, 5):
        r_tmp = np.sqrt((d/2) ** 2 - y ** 2)
        thetas = np.linspace(0, 2 * np.pi, int(r_tmp / 1.42))
        for theta in thetas:
            x = r_tmp * np.cos(theta)
            z = r_tmp * np.sin(theta)
            print(x, y, z)
            Atom(np.array([x, y, z]),
                 np.zeros(3),
                 np.zeros(3),
                 f'capper_{x}{y}{z}')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for tag, atom in Atom.dic_obj.items():
        ax.scatter(*atom.r, color='g')

    set_aspect_equal_3d(ax)

    def update(num):
        print(num)
        if num:
            plt.cla()

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

        for tag_, atom_ in Atom.dic_obj.items():
            atom_.move()
            ax.scatter(*atom_.r, color='g')
            atom_.update()

        ax.set_title(f'{n}, {m} CNT ({num})')

    ani = animation.FuncAnimation(fig, update, interval=100, frames=50)
    ani.save('./test.gif', writer='pillow')


if __name__ == '__main__':
    main(4, 2)
