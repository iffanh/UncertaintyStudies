import numpy as np
import casadi as ca
import chaospy

import lagrange_polynomial as lp

# def f(x:np.ndarray, a:float, b:float): # Assume long running process
#     return np.abs(a*x[0] + b*np.exp(-x[1]))

max_iter = 10
# def f(x:np.ndarray, a, b) -> np.ndarray:
#     return a*(x[0]**2)*(1 + 0.75*np.cos(70*x[0])/12) + np.cos(100*x[0])**2/24 + b*(x[1]**2)*(1 + 0.75*np.cos(70*x[1])/12) + np.cos(100*x[1])**2/24 + 4*x[0]*x[1]
# def f(x:np.ndarray, a) -> np.ndarray:
#     return (x[0]**2) + (x[1]**2) + a*x[0]*x[1]

def f(x:np.ndarray, a) -> np.ndarray: 
    return (x[0] - a)**2 + (x[1] - a)**2 #corresponds to a - Uniform[0,2] E[f(x)] = g(x) = (x[0] - 1)^2 + (x[1] - 1)^2 + 2/3

center = np.array([[10., 10.]]).T
radius = 1.

alpha = chaospy.Uniform(0, 2)
joint = chaospy.J(alpha)
gauss_quads = chaospy.generate_quadrature(20, joint, rule="gaussian") #(order of quadrature, joint prob, rule)
nodes, weights = gauss_quads
expansion = chaospy.generate_expansion(15, joint)
    
for no_iter in range(max_iter):
    
    print(f"============={no_iter}===============")
    if radius < 1e-8:
        print(f"Solution found. x = {center}")
        break
    
    samples = radius*(np.array([[0.524, 0.0006], 
                                [0.032,0.323],
                                [0.187, 0.890],
                                [0.5,0.5],
                                [0.982, 0.368],
                                [0.774,0.918]]).T - \
                                np.array([[0.5, 0.5]]).T) + \
                                center      

    ## Specify uncertainty, quadratures, and PC expansion
    # alpha = chaospy.Normal(10, 2.0)
    # beta = chaospy.Uniform(1.5, 2.5)


    # Build Lagrange polynomials for decision variables
    input_symbols = ca.SX.sym('x', samples.shape[0])
    lag_polys = lp.LagrangePolynomials(input_symbols=input_symbols, pdegree=2)
    lag_polys.initialize(samples)
    lpe = lag_polys.lagrange_polynomials

    ## Build PCE for uncertainty

    model = ca.SX(0)
    model_db = dict()
    for i, x in enumerate(samples.T): # for each control variable, make uncertainty surrogate
        gauss_evals = np.array([f(x, node[0]) for node in nodes.T])
        gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals)
        gauss_model_approx_evals = np.array([gauss_model_approx(node[0]) for node in nodes.T])
        expected = chaospy.E(gauss_model_approx, joint)
        std = chaospy.Std(gauss_model_approx, joint)
        g = 1.0
        robustness = g*expected + (1-g)*std
        model = model + robustness*lpe[i].symbol

    model_approx = ca.Function('approx', [input_symbols], [model])
    center = samples[:, [3]]
    
    # construct NLP problem
    nlp = {
        'x': input_symbols,
        'f': model,
        'g': ca.norm_2(input_symbols - center)
    }
    
    ubg = [radius]
    lbg = [0]
    opts = {'ipopt.print_level':2, 'print_time':2}
    # opts = {}

    # solve TRQP problem
    solver = ca.nlpsol('TRQP_composite', 'ipopt', nlp, opts)
    sol = solver(x0=center+(radius/1E+8), ubg=ubg, lbg=lbg)
    # sol = solver(x0=center, ubg=ubg, lbg=lbg)

    x_new = sol['x'].full()
    evals_new = np.sum(np.array([f(x_new, node[0])*weight for node, weight in zip(nodes.T, weights)]))
    evals_old = np.sum(np.array([f(center, node[0])*weight for node, weight in zip(nodes.T, weights)]))
    evals_approx_new = model_approx(x_new)
    evals_approx_old = model_approx(center)
    
    rho = (evals_new - evals_old)/(evals_approx_new - evals_approx_old)
    
    
    print(f"center (incumbent solution)= {center.T}")
    print(f"x_sol = {x_new.T}")
    print(f"evals_old (incumbent feval) = {evals_old}")
    print(f"evals_new = {evals_new}")
    print(f"evals_approx_old = {evals_approx_old}")
    print(f"evals_approx_new = {evals_approx_new}")
    
    if evals_approx_new > evals_approx_old:
        raise Exception(f"This must not happen. Delta approx = {evals_approx_new - evals_approx_old}. Delta actual = {evals_new - evals_old}")
    
    if rho > 0.4: # good
        center = x_new*1
        radius = radius*1.2
    elif rho > 0.1: 
        center = x_new*0.7
    else:
        radius = radius*0.5
        
    
    print(f"model eval = {sol['f']}")
    print(f"radius = {radius}")
    print(f"rho = {rho}")
    print(f"solver.stats()['success'] = {solver.stats()['success']}")
        
        
## Build surface, one for each sample model.
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from matplotlib import ticker, cm
# fsize = 10
# tsize = 18
# tdir = 'in'
# major = 5.0
# minor = 3.0
# style = 'default'
# plt.style.use(style)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = fsize
# plt.rcParams['legend.fontsize'] = tsize
# plt.rcParams['xtick.direction'] = tdir
# plt.rcParams['ytick.direction'] = tdir
# plt.rcParams['xtick.major.size'] = major
# plt.rcParams['xtick.minor.size'] = minor
# plt.rcParams['ytick.major.size'] = major
# plt.rcParams['ytick.minor.size'] = minor
# # make grid
# X, Y = np.meshgrid(np.linspace(-3, 3, 81),
#                     np.linspace(-3, 3, 101))
# gridsize = X.shape[0]*X.shape[1] 
# surfaces = model_approx.map(gridsize)(ca.horzcat(X.reshape(gridsize), Y.reshape(gridsize)).T)

# Xflat = X.flatten()
# Yflat = Y.flatten()

# surf = []
# for k in range(gridsize):
#     surf.append(model_approx([Xflat[k], Yflat[k]]).full()[:,0])
# surf = np.array(surf) #.reshape((X.shape[0], X.shape[1]))


# levels = [10**i for i in np.arange(-1.0,2.0,0.1)]

# for j in range(model.shape[0]):
#     # surface = surfaces[j, :].reshape((X.shape[0], X.shape[1]))
#     surface = surf[:,j].reshape((X.shape[0], X.shape[1]))

#     fig, ax = plt.subplots(1, 3, figsize=(18, 6))
#     ax[0].set_xlabel(f"$x_1$")
#     ax[0].set_ylabel(f"$x_2$")
#     CS = ax[0].contour(X, Y, surface, levels, norm = LogNorm(), cmap=cm.PuBu_r)
#     ax[0].clabel(CS, fontsize=9, inline=True)
#     ax[0].scatter(samples[0, :], samples[1, :], color='black')
#     # plt.scatter(x[0], y[0], color='red', label=f'Best point')
    
#     act_surface = np.array([f([xv, yv], nodes[0, j], nodes[1, j]) for xv, yv in zip(X.reshape(gridsize), Y.reshape(gridsize))]).reshape((X.shape[0], X.shape[1]))
#     ax[1].set_title(f"$\\alpha$ = {nodes[0, j]}, \n $\\beta = {nodes[1, j]}$")
#     ax[1].set_xlabel(f"$x_1$")
#     ax[1].set_ylabel(f"$x_2$")
#     CS = ax[1].contour(X, Y, act_surface, levels, norm = LogNorm(), cmap=cm.PuBu_r)
#     ax[1].clabel(CS, fontsize=9, inline=True)
#     ax[1].scatter(samples[0, :], samples[1, :], color='black')
    
    
#     center = np.array([0.,0.])
#     radius = 1.0
#     intx = X - center[0]
#     inty = Y - center[1]
#     dist = np.sqrt(intx**2 + inty**2)
#     intindices = dist <= radius
#     clipped_surf = act_surface*1
#     clipped_surf[intindices] = surface[intindices]
#     circle1 = plt.Circle(center, radius, color='black', fill=False)
#     CS = ax[2].contour(X, Y, clipped_surf, levels, norm = LogNorm(), cmap=cm.PuBu_r)
#     ax[2].add_patch(circle1)
#     ax[2].scatter(samples[0, :], samples[1, :], color='black')
    
#     plt.savefig(f'./PCE/surfaces/surface_{j}')
#     plt.close()