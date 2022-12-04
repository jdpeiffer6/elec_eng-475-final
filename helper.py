import sympy as sym
import numpy as np
from matplotlib import pyplot as plt

def animate_single_pend(theta_array,L1=1,T=10):
    """
    Function to generate web-based animation of double-pendulum system

    Parameters:
    ================================================
    theta_array:
        trajectory of theta1 and theta2, should be a NumPy array with
        shape of (2,N)
    L1:
        length of the first pendulum
    T:
        length/seconds of animation duration

    Returns: None
    """

    ################################
    # Imports required for animation.
    from plotly.offline import init_notebook_mode, iplot
    from IPython.display import display, HTML
    import plotly.graph_objects as go

    #######################
    # Browser configuration.
    def configure_plotly_browser_state():
        import IPython
        display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            '''))
    configure_plotly_browser_state()
    init_notebook_mode(connected=False)

    ###############################################
    # Getting data from pendulum angle trajectories.
    xx1=L1*np.sin(theta_array[0])
    yy1=-L1*np.cos(theta_array[0])
    N = len(theta_array[0]) # Need this for specifying length of simulation

    ####################################
    # Using these to specify axis limits.
    xm=np.min(xx1)-0.5
    xM=np.max(xx1)+0.5
    ym=np.min(yy1)-2.5
    yM=np.max(yy1)+1.5

    ###########################
    # Defining data dictionary.
    # Trajectories are here.
    data=[dict(x=xx1, y=yy1, 
               mode='lines', name='Arm', 
               line=dict(width=2, color='blue')
              ),
          dict(x=xx1, y=yy1, 
               mode='lines', name='Mass 1',
               line=dict(width=2, color='purple')
              ),
          dict(x=xx1, y=yy1, 
               mode='markers', name='Pendulum 1 Traj', 
               marker=dict(color="purple", size=2)
              ),
        ]

    ################################
    # Preparing simulation layout.
    # Title and axis ranges are here.
    layout=dict(xaxis=dict(range=[xm, xM], autorange=False, zeroline=False,dtick=1),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False,scaleanchor = "x",dtick=1),
                title='Double Pendulum Simulation', 
                hovermode='closest',
                updatemenus= [{'type': 'buttons',
                               'buttons': [{'label': 'Play','method': 'animate',
                                            'args': [None, {'frame': {'duration': T, 'redraw': False}}]},
                                           {'args': [[None], {'frame': {'duration': T, 'redraw': False}, 'mode': 'immediate',
                                            'transition': {'duration': 0}}],'label': 'Pause','method': 'animate'}
                                          ]
                              }]
               )

    ########################################
    # Defining the frames of the simulation.
    # This is what draws the lines from
    # joint to joint of the pendulum.
    frames=[dict(data=[dict(x=[0,xx1[k]], 
                            y=[0,yy1[k]], 
                            mode='lines',
                            line=dict(color='red', width=3)
                            ),
                       go.Scatter(
                            x=[xx1[k]],
                            y=[yy1[k]],
                            mode="markers",
                            marker=dict(color="blue", size=12)),
                      ]) for k in range(N)]

    #######################################
    # Putting it all together and plotting.
    figure1=dict(data=data, layout=layout, frames=frames)           
    iplot(figure1)
    
    
def integrate(f, xt, dt, state_length):
    """
    This function takes in an initial condition x(t) and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x(t). It outputs a vector x(t+dt) at the future
    time step.
    
    Parameters
    ============
    dyn: Python function
        derivate of the system at a given step x(t), 
        it can considered as \dot{x}(t) = func(x(t))
    xt: NumPy array
        current step x(t)
    dt: 
        step size for integration

    Return
    ============
    new_xt: 
        value of x(t+dt) integrated from x(t)
    """
    k1 = dt * f(xt[:state_length])
    k2 = dt * f(xt[:state_length]+k1[:state_length]/2.)
    k3 = dt * f(xt[:state_length]+k2[:state_length]/2.)
    k4 = dt * f(xt[:state_length]+k3[:state_length])
    new_xt = xt[:state_length] + (1/6.) * (k1[:state_length]+2.0*k2[:state_length]+2.0*k3[:state_length]+k4[:state_length])
    
    return np.concatenate([new_xt, f(xt[:state_length])[state_length:]])

def simulate(f, x0, tspan, dt, integrate):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    
    Parameters
    ============
    f: Python function
        derivate of the system at a given step x(t), 
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        initial conditions
    tspan: Python list
        tspan = [min_time, max_time], it defines the start and end
        time of simulation
    dt:
        time step for numerical integration
    integrate: Python function
        numerical integration method used in this simulation

    Return
    ============
    x_traj:
        simulated trajectory of x(t) from t=0 to tf
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0) + 1,N))
    for i in range(N):
        xtraj[:,i]=integrate(f, x, dt, len(x0))
        x = np.copy(xtraj[:len(x0),i])
    return xtraj 

t, g, m, R = sym.symbols(r't, g, m, R')
th = sym.Function(r'\theta')(t)
tau = sym.Function(r'\tau')(t)
thdot = th.diff((t,1))
thddot = thdot.diff((t,1))

Lagrangian = 0.5*m*(R*thdot)**2 - m*g*R*(1-sym.cos(th))
q = sym.Matrix([th])
qdot = sym.Matrix([thdot])
L = sym.Matrix([Lagrangian])
dL_dq = L.jacobian(q)
dL_dqdot = L.jacobian(qdot)
ddt_dL_dqdot = []
for term in dL_dqdot:
    ddt_dL_dqdot.append(term.diff(t))
ddt_dL_dqdot = sym.Matrix([ddt_dL_dqdot])

EL = dL_dq - ddt_dL_dqdot
EL = EL.T # express EL as a coln vector
EL_eqn = sym.Eq(sym.simplify(ddt_dL_dqdot - dL_dq), sym.Matrix([tau]))
print('Euler-Lagrange Equation:')
display(sym.simplify(EL_eqn))

a = sym.Matrix([thddot])
EL_soln = sym.solve(EL_eqn, a, dict=True)
for i in range(len(EL_soln)):
    EL_soln[i] = sym.simplify(EL_soln[i])

func = []
for sol in EL_soln:
    print('\nSolution of EL eqn in variables: ')
    for i in a:
        display(sym.simplify(sym.Eq(i, sol[i])))
        func.append(sol[i])

# substitute constants into symbolic solutions
subs_dict = {m:1, g:9.8, R:1}
thddot_sol = EL_soln[0][thddot].subs(subs_dict)
thddot_func = sym.lambdify([th, thdot, tau], thddot_sol)
# define extended dynamics for the cart pendulum system
def dyn(s):
  # TODO: Change tau_in for whatever control is desired
  tau_in = -m*g*R*sym.sin(s[0])
  # return np.float64(np.array([s[1], thddot_func(s[0], s[1], tau_in.subs(subs_dict)), tau_in.subs(subs_dict)]))
  return np.float64(np.array([s[1], thddot_func(s[0], s[1], 0.), 0.]))
# initial conditions
x0 = np.array([np.pi/2, 0])
# simulate the trajectory
T = 5.
dt = 0.01

# traj = simulate(dyn, x0, [0, T], dt, integrate)