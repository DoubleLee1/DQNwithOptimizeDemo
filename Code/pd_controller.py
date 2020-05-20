class PID_controller :
    def __init__(self, p, d, i):
        self.Kp = p
        self.Kd = d
        self.Ki = d
        self.p_error = 0
        self.d_error = 0
        self.i_error = 0
        self.prev_error = 0

    def update(self, cte):

        error = cte
        self.p_error = error
        self.d_error = error - self.prev_error
        self.i_error += error

        output = self.Kp*self.p_error + self.Kd*self.d_error + self.Ki*self.i_error

        if output > 1:
            output = 1
        elif output < -1:
            output = -1

        return output