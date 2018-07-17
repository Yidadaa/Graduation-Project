import tensorflow as tf
import numpy as np

def sqrt_of_two(a, b):
    return tf.pow(add(a, tf.negative(b)), 2)

def add(a, b):
    return tf.add(a, b)

def adds(list_of_e):
    s = tf.zeros(1)
    for e in list_of_e:
        s = add(s, e)
    return s

class Decide():
    sess = tf.Session()

    alpha = tf.placeholder(tf.float32)
    beta = tf.placeholder(tf.float32)
    theta = tf.placeholder(tf.float32)

    dx = tf.placeholder(tf.float32)
    dy = tf.placeholder(tf.float32)
    dz = tf.placeholder(tf.float32)

    l1 = tf.placeholder(tf.float32)
    l2 = tf.placeholder(tf.float32)
    l3 = tf.placeholder(tf.float32)

    ll = add(tf.multiply(l2, tf.sin(beta)), tf.multiply(l3, tf.sin(theta)))

    px = tf.multiply(ll, tf.cos(alpha))
    py = tf.multiply(ll, tf.sin(alpha))
    pz = add(l1, add(tf.multiply(l2, tf.cos(beta)), tf.multiply(l3, tf.cos(theta))))

    f = tf.sqrt(adds([sqrt_of_two(px, dx), sqrt_of_two(py, dy), sqrt_of_two(pz, dz)]))

    df_dalpha = tf.gradients(f, alpha)
    df_dbeta = tf.gradients(f, beta)
    df_dtheta = tf.gradients(f, theta)

    def get_delta_pos(self, angles, target, arm):
        angles = np.array(angles) / 180 * np.pi
        feed_dict = {
            self.alpha: angles[0], self.beta: angles[1], self.theta: angles[2],
            self.dx: target[0], self.dy: target[1], self.dz: target[2],
            self.l1: arm[0], self.l2: arm[1], self.l3: arm[2]
        }
        return [self.sess.run(op, feed_dict=feed_dict) for op in [self.df_dalpha, self.df_dbeta, self.df_dtheta, self.f]]

    def test(self):
        a = tf.constant(1, tf.float32)
        b = tf.constant(2, tf.float32)
        c = tf.constant(1, tf.float32)
        test_sqrt_of_two = sqrt_of_two(a, b)
        test_add = add(b, c)
        test_adds = adds([a, b, c])
        res = [self.sess.run(op) for op in [test_sqrt_of_two, test_add, test_adds]]
        return res

def run():
    Brain = Decide()
    angles = np.array([0, 90, 90])
    target = [2, 3, 0]
    l = [1, 1, 1]
    s = np.zeros(3)
    count = 0
    lastdis = 1000000
    mu = 0.5
    lastdeltas = None
    while True:
        count = count + 1
        res = Brain.get_delta_pos(angles, target, l)
        dis = res[-1][0]
        deltas = np.array(res[0:3]).reshape(3) * 0.1
        if count > 1:
            deltas = (1 - mu) * lastdeltas + deltas
        lastdeltas = deltas
        if dis < 0.1 or lastdis - dis < 1e-5 :
            break
        else:
            angles = angles - deltas
            s = s + deltas
            print(str(count) + ': ' + str(angles) + '  ' + str(dis))
            lastdis = dis
    print(s)

if __name__ == '__main__':
    run()