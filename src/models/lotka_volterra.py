from scipy.integrate import solve_ivp, RK45, DOP853


def _lotka_volterra(t, state, alpha, beta, gamma, delta):
    # expecting 2-dim list
    if len(state) != 2:
        raise Exception("Expected 2-dimensional initial state: [x, y]")

    x = state[0]
    y = state[1]

    # dx/dt
    dx = (alpha*x) - (beta*x*y)
    # dy/dt
    dy = (delta*x*y) - (gamma*y)

    return [dx, dy]


class LotkaVolterra():
    """
    The basic Lotka-Volterra model.
    """
    def __init__(self, init_state, alpha, beta, gamma, delta):
        self.init_state = init_state
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def solve(self, t_start, t_end):
        solution = solve_ivp(
            _lotka_volterra, 
            t_span=[t_start, t_end],
            y0=self.init_state,
            args=(self.alpha, self.beta, self.gamma, self.delta),
            dense_output=True,
            method=DOP853
        )
        return solution