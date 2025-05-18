import random

'''
M. Yajnik, S. Moon, J. Kurose, and D. Towsley, “Measurement and modelling
of the temporal dependence in packet loss,” in IEEE INFOCOM
’99. Conference on Computer Communications. Proceedings. Eighteenth
Annual Joint Conference of the IEEE Computer and Communications
Societies., vol. 1, 1999, pp. 345–352 vol.1.
'''


class Random_packet_tracer(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
        self.name = 'Random_packet_tracer'

    def generate(self, length) -> str:
        status = [str(int(random.random() > self.drop_rate)) for _ in range(length)]
        return ''.join(status)  # 0 for loss, 1 for good


class Two_state_Markov_wlan_packet_tracer(object):
    def __init__(self):
        self.name = 'Two_state_Markov_wlan_packet_tracer'
        self.Q = 0.9042  # state 1 self-loop
        self.q = 0.4060  # prob of consecutive packet loss

        self.P = 1 - self.Q  # state 1 to 2(loss)
        self.p = 1 - self.q  # return to no loss
        self.state = random.randint(1, 2)  # state 1, 3: good state no packet loss  # state 2 : packet loss

    def avg_packet_loss_rate(self):
        return self.P / (2 - self.Q - self.q)

    def avg_packet_loss_burst_loss_length(self):
        return 1. / (1 - self.q)

    def next_state(self):
        if self.state == 1:
            self.state += (random.uniform(0, 1) < self.P)
        elif self.state == 2:
            rng = random.uniform(0, 1)
            self.state -= (rng < self.p)
            self.state += (rng > self.p + self.q)
        return self.state

    def generate(self, length) -> str:
        status = [str(int(self.next_state() != 2)) for _ in range(length)]
        return ''.join(status)  # 0 for loss, 1 for good


class Three_state_Markov_wlan_packet_tracer(object):
    def __init__(self, mode='wlan'):
        self.name = '[] Three_state_Markov_wlan_packet_tracer'.format(mode)

        if mode == 'EP1':
            self.Q = 0.99968  # state 1 self-loop
            self.P = 1 - self.Q  # state 1 to 2(loss)
            self.Qprime = 0  # state 3 self-loop
            self.p = 0.1538  # return to no loss
            self.q = 0.8462  # prob of consecutive packet loss

        elif mode == 'EP2':
            self.Q = 0.9798  # state 1 self-loop
            self.P = 1 - self.Q  # state 1 to 2(loss)
            self.Qprime = 0.3333  # state 3 self-loop
            self.q = 0.372  # return to no loss
            self.p = 0.6304  # prob of consecutive packet loss

        elif mode == 'EP3':
            self.Q = 0.95  # state 1 self-loop
            self.P = 1 - self.Q  # state 1 to 2(loss)
            self.Qprime = 0.6  # state 3 self-loop
            self.p = 0.8  # return to no loss
            self.q = 0.8  # prob of consecutive packet loss

        elif mode == 'EP4':
            self.Q = 0.9363  # state 1 self-loop
            self.P = 1 - self.Q  # state 1 to 2(loss)
            self.Qprime = 0.5662  # state 3 self-loop

            self.p = 0.3631  # return to no loss
            self.q = 0.4072  # prob of consecutive packet loss

        elif mode == 'EP5':
            self.Q = 0.9  # state 1 self-loop
            self.P = 1 - self.Q  # state 1 to 2(loss)
            self.Qprime = 0.1  # state 3 self-loop
            self.p = 0.4  # return to no loss
            self.q = 0.9  # prob of consecutive packet loss

        elif mode == 'EP6':
            self.Q = 0.8507
            self.P = 1 - self.Q
            self.p = 0.2982
            self.Qprime = 0.2000
            self.q = 0.6305
        self.state = random.randint(1, 3)  # state 1, 3: good state no packet loss  # state 2 : packet loss

    def avg_packet_loss_rate(self):
        return (1 - self.Qprime) * (1 - self.Q) / (
                self.Q * (self.p + self.q + self.Qprime - 2) - self.Qprime * (1 + self.p) - self.q + 2)

    def avg_packet_loss_burst_loss_length(self):
        return 1. / (1 - self.q)

    def next_state(self):
        if self.state == 1:
            self.state += (random.uniform(0, 1) < self.P)
        elif self.state == 2:
            rng = random.uniform(0, 1)
            self.state -= (rng < self.p)
            self.state += (rng > self.p + self.q)
        else:
            self.state -= (random.uniform(0, 1) > self.Qprime)
        return self.state

    def generate(self, length) -> str:
        status = [str(int(self.next_state() != 2)) for _ in range(length)]
        return ''.join(status)  # 0 for loss, 1 for good


if __name__ == '__main__':
    for case in ['EP1', 'EP2', 'EP3', 'EP4', 'EP5', 'EP6']:
        model = Three_state_Markov_wlan_packet_tracer(case)
        print('{}, Avg packet loss rate: {}, Avg burst loss length: {}'.format(
            case, model.avg_packet_loss_rate(), model.avg_packet_loss_burst_loss_length()))

        # Monte Carlo Simulation
        n = 10000000
        status = model.generate(n)
        loss = status.count('0')
        print('Loss rate from Monte Carlo Simulation: ', loss / n)

