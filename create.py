import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path = './data/'

aC_C = 1.42
a = np.sqrt(3) * aC_C
a1 = np.array([np.sqrt(3) / 2, 1 / 2]) * a
a2 = np.array([np.sqrt(3) / 2, -1 / 2]) * a


class Line:
    def __init__(self, x0, y0, x1, y1, tag):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.tag = tag
        self.points = [np.array([x0, x1]), np.array([y0, y1])]

    def if_inside(self, x, y):
        vec_edge = np.array([self.x1 - self.x0, self.y1 - self.y0])
        vec_point = np.array([x - self.x0, y - self.y0])
        cross = np.cross(vec_edge, vec_point)
        if cross > 0:
            return True
        else:
            return False

    def if_onside(self, x, y):
        e = 0.00001
        vec_edge = np.array([self.x1 - self.x0, self.y1 - self.y0])
        vec_point = np.array([x - self.x0, y - self.y0])
        d = np.cross(vec_edge, vec_point) / np.linalg.norm(vec_edge)
        if abs(d) < e and (self.x0 - e < x < self.x1 + e or self.x1 - e < x < self.x0 + e):
            return True
        else:
            return False


class C:
    C_dict = {}
    C_dict_rolled = {}

    def __init__(self, x, y, z=0, tag=None):
        self.x = x
        self.y = y
        self.z = z
        self.__tag = tag

    @property
    def tag(self):
        return self.__tag

    @tag.setter
    def tag(self, tag):  # タグを設定するとdictに追加される
        self.__tag = tag
        C.C_dict[self.__tag] = np.array([self.x, self.y, self.z])

    def find_nearest(self, rolled=False):
        if rolled:
            dic = C.C_dict_rolled
        else:
            dic = C.C_dict

        nearest_C_tags = []
        for tag, loc in dic.items():
            if tag == self.tag:
                continue
            my_loc = np.array([self.x, self.y, self.z])
            distance = np.linalg.norm(my_loc - loc)
            if distance < aC_C + 0.1:
                nearest_C_tags.append(tag)
        return nearest_C_tags


class Bond:
    bond_set = set()
    bond_set_rolled = set()

    def __init__(self, tag1, tag2, rolled=False):
        self.cc = tuple(sorted([tag1, tag2]))
        if rolled:
            Bond.bond_set_rolled.add(self.cc)
        else:
            Bond.bond_set.add(self.cc)


def roll(n, m, Cs):
    Ch = n * a1 + m * a2
    phi = np.arctan(Ch[1] / Ch[0])
    L = np.linalg.norm(Ch)
    d = L / np.pi

    R = np.array([
        [np.cos(phi), np.sin(phi), 0],
        [-1 * np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    for c in Cs:
        x, y, z = R @ np.array([c.x, c.y, c.z])
        rolled_x = d / 2 * np.cos(x / (d / 2))
        rolled_z = d / 2 * np.sin(x / (d / 2))
        c.x, c.y, c.z = rolled_x, y, rolled_z
        C.C_dict_rolled[c.tag] = np.array([rolled_x, y, rolled_z])


def calc_rotation(loc1, loc2, show=False):
    if loc1[2] < loc2[2]:
        tmp = loc1
        loc1 = loc2
        loc2 = tmp

    center = (loc1 + loc2) / 2
    length = np.linalg.norm(loc1 - loc2)
    theta_y = np.arccos((loc1[2] - center[2]) / (length / 2))

    before1 = center + np.array([0, 0, length / 2])
    before2 = center - np.array([0, 0, length / 2])

    mid1 = rotate((before1 - center), [0, theta_y, 0]) + center
    mid2 = rotate((before2 - center), [0, theta_y, 0]) + center

    theta_loc = np.arctan2(loc1[1] - center[1], loc1[0] - center[0])
    theta_mid = np.arctan2(mid1[1] - center[1], mid1[0] - center[0])
    theta_z = theta_loc - theta_mid

    after1 = rotate((mid1 - center), [0, 0, -theta_z]) + center
    after2 = rotate((mid2 - center), [0, 0, -theta_z]) + center

    # print(f'角度誤差：{np.linalg.norm(after1 - loc1)}')

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([center[0], center[0] + np.cos(theta_loc)], [center[1], center[1] + np.sin(theta_loc)], [0, 0], color='k')
        ax.plot([center[0], center[0] + np.cos(theta_mid)], [center[1], center[1] + np.sin(theta_mid)], [0, 0], color='k')
        ax.scatter(*loc1, color='k')
        ax.text(*loc1, 'loc1')
        ax.scatter(*loc2, color='k')
        ax.text(*loc2, 'loc2')
        ax.scatter(*before1, color='b')
        ax.text(*before1, 'before1')
        ax.scatter(*before2, color='b')
        ax.text(*before2, 'before2')
        ax.scatter(*mid1, color='g')
        ax.text(*mid1, 'mid1')
        ax.scatter(*mid2, color='g')
        ax.text(*mid2, 'mid2')
        ax.scatter(*after1, color='r')
        ax.text(*after1, 'after1')
        ax.scatter(*after2, color='r')
        ax.text(*after2, 'after2')
        plt.show()

    return -theta_y, theta_z


def rotate(loc, angle):
    px = angle[0]
    py = angle[1]
    pz = angle[2]

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(px), np.sin(px)],
                   [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                   [0, 1, 0],
                   [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                   [-np.sin(pz), np.cos(pz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    return R @ loc


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


def main(n, m, l, show=False):
    # 初期化
    C.C_dict = {}
    C.C_dict_rolled = {}
    Bond.bond_set = set()
    Bond.bond_set_rolled = set()

    Ch = n * a1 + m * a2
    dR = np.gcd(n, m)
    t1 = int((2 * m + n) / dR)
    t2 = int(- (2 * n + m) / dR)
    T = t1 * a1 + t2 * a2
    T = T * l
    N = int(2 * (n ** 2 + m ** 2 + n * m) / dR)

    if show:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(223)
        ax_3d = fig.add_subplot(122, projection='3d')
        ax.set_title(f'({n}, {m})')
        ax.set_aspect('equal')
        ax2.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

    lines = [Line(0, 0, Ch[0], Ch[1], 'bottom'),
             Line(Ch[0], Ch[1], (T+Ch)[0], (T+Ch)[1], 'right'),
             Line((T+Ch)[0], (T+Ch)[1], T[0], T[1], 'top'),
             Line(T[0], T[1], 0, 0, 'left')]

    Cs_tmp = []
    for i in np.arange(0 * l, (n + t1 + 1) * l):
        for j in np.arange(t2 * l, (m + 1) * l):
            tmp = C(*(i * a1 + j * a2))
            Cs_tmp.append(tmp)
            tmp = C(*(i * a1 + j * a2 + np.array([aC_C, 0])))
            Cs_tmp.append(tmp)

    Cs = []
    tag_left = tag_right = tag_in = 0
    for i, c in enumerate(Cs_tmp):
        count = 0
        for line in lines:
            if line.if_onside(c.x, c.y) and line.tag in ['left', 'right']:  # 辺上のものは追加
                count = 0
                if line.tag == 'left':  # T上
                    c.tag = f'C_left_{tag_left}'
                    tag_left += 1
                    Cs.append(c)
                else:  # Ch上
                    # c.tag = f'C_right_{tag_right}'
                    tag_right += 1
                break
            if line.if_inside(c.x, c.y):
                count += 1
        if count == 4:
            c.tag = f'C_in_{tag_in}'
            tag_in += 1
            Cs.append(c)

    for c in Cs:
        nearest_C_tags = c.find_nearest()
        for nearest_C_tag in nearest_C_tags:
            Bond(c.tag, nearest_C_tag)

    if show:
        for line in lines:
            ax.plot(*line.points, color='k')

        for tag, loc in C.C_dict.items():
            ax.text(loc[0], loc[1], tag)
            ax.scatter(loc[0], loc[1], color='g')

        for bond in Bond.bond_set:
            x0, y0, z0 = C.C_dict[bond[0]]
            x1, y1, z1 = C.C_dict[bond[1]]
            ax.plot([x0, x1], [y0, y1], color='b')

    roll(n, m, Cs)

    for c in Cs:
        nearest_C_tags = c.find_nearest(rolled=True)
        for nearest_C_tag in nearest_C_tags:
            Bond(c.tag, nearest_C_tag, rolled=True)

    if show:
        for tag, loc in C.C_dict_rolled.items():
            ax2.text(loc[0], loc[1], tag)
            ax2.scatter(loc[0], loc[1], color='g')

            ax_3d.text(*loc, tag, size=5)
            ax_3d.scatter(*loc, color='g')

        for bond in Bond.bond_set_rolled:
            x0, y0, z0 = C.C_dict_rolled[bond[0]]
            x1, y1, z1 = C.C_dict_rolled[bond[1]]
            ax_3d.plot([x0, x1], [y0, y1], [z0, z1], color='b')

    df_C = pd.DataFrame(C.C_dict_rolled)
    df_C.index = ['x', 'y', 'z']
    df_B = None
    for bond in Bond.bond_set_rolled:
        loc1, loc2 = df_C[bond[0]].values, df_C[bond[1]].values
        center = (loc1 + loc2) / 2
        theta_y, theta_z = calc_rotation(loc1, loc2)
        df_tmp = pd.DataFrame(data=[*center, 0, theta_y, theta_z],
                              index=['x', 'y', 'z', 'theta_x', 'theta_y', 'theta_z'],
                              columns=['B_' + '-'.join(bond)])
        if df_B is None:
            df_B = df_tmp
        else:
            df_B = pd.concat([df_B, df_tmp], axis=1)

    df_C.T.to_csv(f'{output_path}/atoms_{n}_{m}.csv')
    df_B.T.to_csv(f'{output_path}/bonds_{n}_{m}.csv')

    print(f'({n}, {m}) data has been created.')

    if show:
        set_aspect_equal_3d(ax_3d)
        plt.show()


if __name__ == '__main__':
    for n_ in range(1, 10):
        for m_ in range(0, n_ + 1):
            main(n_, m_, 2)
