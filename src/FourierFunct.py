import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

#--------------------------------------------------------------
# Transform a function into periodic.
#--------------------------------------------------------------
def f_periodic(func, x, A=-np.pi, B=np.pi):
    """Makes func periodic over [A, B]."""
    L = B - A
    # Traslada x para centrar el intervalo en [A, B]
    x_mod = ((x - A) % L) + A
    return func(x_mod)

def animate_fourier(func, filename, N_MAX=30, A=-np.pi, B=np.pi, M=2000, fps=5, h_max=1.5, h_min=1.5):
    """
Creates an animation showing how a function 'func' is approximated
by its Fourier series (sine and cosine terms) over the interval [A, B].

Parameters
----------
func : callable
    The target function f(x) to approximate. Must accept NumPy arrays as input.
filename : str
    Name of the output file. Must end with .gif.
N_MAX : int
    Maximum number of Fourier series terms to include in the animation. Default is 30.
A, B : float
    Lower and upper bounds of the fundamental interval. Defaults are -π and π.
M : int, optional
    Number of sampling points for numerical integration and plotting. Default is 2000.
fps : int, optional
    Frames per second for the output animation. Default is 5.
h_max, h_min : float, optional
    Vertical scaling factors for the plot limits. Default is 1.5 for both.

Returns
-------
ani : matplotlib.animation.FuncAnimation
    The generated animation object, which can be displayed or saved.
    """

    # Interval length
    L = B - A
    x = np.linspace(A, B, M)


    # --------------------------------------------------------------
    # Calculate Fourier coefficients
    # --------------------------------------------------------------
    fx = func(x)
    factor = 2.0 / L
    a0 = factor * np.trapezoid(fx, x)

    an = np.zeros(N_MAX)
    bn = np.zeros(N_MAX)
    for n in range(1, N_MAX + 1):
        an[n - 1] = factor * np.trapezoid(fx * np.cos(n * np.pi * (x - A) / (L / 2)), x)
        bn[n - 1] = factor * np.trapezoid(fx * np.sin(n * np.pi * (x - A) / (L / 2)), x)

    # --------------------------------------------------------------
    # Partial sums
    # --------------------------------------------------------------
    def fourier_partial_sum(x_eval, a0, an, bn, n_terms):
        s = np.full_like(x_eval, a0 / 2.0)
        for k in range(1, n_terms + 1):
            s += an[k - 1] * np.cos(k * np.pi * (x_eval - A) / (L / 2))
            s += bn[k - 1] * np.sin(k * np.pi * (x_eval - A) / (L / 2))
        return s

    # --------------------------------------------------------------
    # Prepare animation
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(A, B)

    # Set y-limits based on function values and scaling factors
    y_min, y_max = np.min(fx), np.max(fx)
    if y_min == 0:
        y_min = -1 #Avoid zero range
    elif y_min > 0:
        y_min = 0
    if y_max == 0:
        y_max = 1 #Avoid zero range
    elif y_max < 0:
        y_max = 0
    if y_min == y_max:
        y_max = y_min +1
        y_min -= 1
    ax.set_ylim(h_min * y_min, h_max * y_max)



    ax.set_xlabel("x")
    ax.set_ylabel("f(x) y Sₙ(x)")
    ax.set_title("Aproximación por Serie de Fourier")

    line_f, = ax.plot(x, fx, 'k--', linewidth=1, label='f(x)')
    line_sn, = ax.plot(x, fourier_partial_sum(x, a0, an, bn, 1), 'r', linewidth=2, label='Sₙ(x)')
    text_n = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def update(frame):
        n = frame
        y = fourier_partial_sum(x, a0, an, bn, n)
        line_sn.set_data(x, y)
        text_n.set_text(f'n = {n}')
        return line_sn, text_n

    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(1, N_MAX + 1), interval=800, blit=True
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Save animation
    # --------------------------------------------------------------
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_dir, filename)

    if filename.endswith(".gif"):
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        print("⚠️ Formato no reconocido. Usa .gif")
        return ani

    print(f"✅ Animación guardada en: {output_path}")
    return ani


#--------------------------------------------------------------
# Now we will define same function but with Frejer's sums
#--------------------------------------------------------------

def animate_frejer(func, filename, N_MAX=30, A=-np.pi, B=np.pi, M=2000, fps=5, h_max=1.5, h_min=1.5):

    """
Creates an animation showing how a function 'func' is approximated
by its Frejer series (sine and cosine terms) over the interval [A, B].

Parameters
----------
func : callable
    The target function f(x) to approximate. Must accept NumPy arrays as input.
filename : str
    Name of the output file. Must end with .gif.
N_MAX : int
    Maximum number of Fourier series terms to include in the animation. Default is 30.
A, B : float
    Lower and upper bounds of the fundamental interval. Defaults are -π and π.
M : int, optional
    Number of sampling points for numerical integration and plotting. Default is 2000.
fps : int, optional
    Frames per second for the output animation. Default is 5.
h_max, h_min : float, optional
    Vertical scaling factors for the plot limits. Default is 1.5 for both.

Returns
-------
ani : matplotlib.animation.FuncAnimation
    The generated animation object, which can be displayed or saved.
    """

    # Interval length
    L = B - A
    x = np.linspace(A, B, M)


    # --------------------------------------------------------------
    # Calculate Frejer coefficients
    # --------------------------------------------------------------
    fx = func(x)
    factor = 2.0 / L
    a0 = factor * np.trapezoid(fx, x)

    an = np.zeros(N_MAX)
    bn = np.zeros(N_MAX)
    for n in range(1, N_MAX + 1):
        an[n - 1] = factor * np.trapezoid(fx * np.cos(n * np.pi * (x - A) / (L / 2)), x)
        bn[n - 1] = factor * np.trapezoid(fx * np.sin(n * np.pi * (x - A) / (L / 2)), x)

    # --------------------------------------------------------------
    # Partial sums
    # --------------------------------------------------------------
    def frejer_partial_sum(x_eval, a0, an, bn, n_terms):
        # Fejér is the average of partial Fourier sums:
        # S_F,N(x) = (1/(N+1)) * sum_{m=0}^N S_m(x)
        # where S_0 = a0/2 and S_m = S_{m-1} + a_m cos(...) + b_m sin(...)
        s_cum = np.zeros_like(x_eval, dtype=float)
        S_m = np.full_like(x_eval, a0 / 2.0, dtype=float)
        s_cum += S_m  # include S_0
        for m in range(1, n_terms + 1):
            S_m = S_m + an[m - 1] * np.cos(m * np.pi * (x_eval - A) / (L / 2)) \
                        + bn[m - 1] * np.sin(m * np.pi * (x_eval - A) / (L / 2))
            s_cum += S_m
        return s_cum / (n_terms + 1)

    # --------------------------------------------------------------
    # Prepare animation
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(A, B)

    # Set y-limits based on function values and scaling factors
    y_min, y_max = np.min(fx), np.max(fx)
    if y_min == 0:
        y_min = -1 #Avoid zero range
    elif y_min > 0:
        y_min = 0
    if y_max == 0:
        y_max = 1 #Avoid zero range
    elif y_max < 0:
        y_max = 0
    if y_min == y_max:
        y_max = y_min +1
        y_min -= 1
    ax.set_ylim(h_min * y_min, h_max * y_max)



    ax.set_xlabel("x")
    ax.set_ylabel("f(x) y σₙ(x)")
    ax.set_title("Frejer sums approximation")

    line_f, = ax.plot(x, fx, 'k--', linewidth=1, label='f(x)')
    line_sn, = ax.plot(x, frejer_partial_sum(x, a0, an, bn, 1), 'r', linewidth=2, label='σₙ(x)')
    text_n = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def update(frame):
        n = frame
        y = frejer_partial_sum(x, a0, an, bn, n)
        line_sn.set_data(x, y)
        text_n.set_text(f'n = {n}')
        return line_sn, text_n

    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(1, N_MAX + 1), interval=800, blit=True
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Save animation
    # --------------------------------------------------------------
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_dir, filename)

    if filename.endswith(".gif"):
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        print("⚠️ Formato no reconocido. Usa .gif")
        return ani

    print(f"✅ Animación guardada en: {output_path}")
    return ani
