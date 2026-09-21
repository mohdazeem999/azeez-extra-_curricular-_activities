import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# constants
k = 8.99e9          # Coulomb constant (N·m²/C²)
e = 1.6e-19         # elementary charge (C)
m_alpha = 6.64e-27  # mass of alpha particle (kg)

# setup
print("=" * 40)
print("    Rutherford scattering")
print("   choose target nucleus")
print("=" * 40)

elements = {
    '1': ('gold', 79),
    '2': ('silver', 47),
    '3': ('copper', 29),
    '4': ('carbon', 6),
    '5': ('platinum', 78),
    '6': ('zinc', 30),
    '7': ('iron', 26),
    '8': ('lead', 82),
    '9': ('uranium', 92),
    '10': ('hydrogen', 1),
    '11': ('aluminium', 13),
}

for key, (n, z) in elements.items():
    print(f" {key:>2}. {n:<12} Z={z}")

print("=" * 40)
choice = input("Enter number: ").strip()

if choice not in elements:
    raise ValueError("Invalid choice.")

name, z = elements[choice]
print(f"\nSimulating {name} (Z={z})...")

# Alpha particle energy ~ 7.7 MeV (typical Rutherford experiment)
E_MeV = 7.7
E_J = E_MeV * 1e6 * e  # convert MeV to joules
v0 = np.sqrt(2 * E_J / m_alpha)

# impact parameters
b_values = np.linspace(1e-15, 8e-14, 30)

def simulate(b, v0, z):
    """
    Simulate one alpha particle trajectory with impact parameter b.
    Returns xs, ys, and a boolean: True if it passes through (vx_f > 0), False if bounced back.
    """
    x = -8e-14
    y = b
    vx = v0
    vy = 0.0
    dt = 5e-23

    xs = [x]
    ys = [y]

    for _ in range(8000):
        r = np.sqrt(x**2 + y**2)
        if r < 1e-17:
            break

        # Coulomb force magnitude: F = k * (2e)(Ze) / r^2
        F = k * (2 * e) * (z * e) / (r**2)
        Fx = F * x / r
        Fy = F * y / r

        ax_a = Fx / m_alpha
        ay_a = Fy / m_alpha

        vx += ax_a * dt
        vy += ay_a * dt
        x += vx * dt
        y += vy * dt

        xs.append(x)
        ys.append(y)

        # stop conditions
        if x > 2e-14:      # exited to the right
            break
        if x < -9e-14:     # bounced far left
            break

    vx_f = vx
    return xs, ys, (vx_f > 0)

# pre-calculate all trajectories
print("Precalculating trajectories ...")
all_traj = []
bounced = 0
total = 0

for b in b_values:
    # positive b
    xs, ys, passed = simulate(b, v0, z)
    all_traj.append((xs, ys, passed))
    # negative b (mirror)
    xs2, ys2, passed2 = simulate(-b, v0, z)
    all_traj.append((xs2, ys2, passed2))

    total += 2
    if not passed:
        bounced += 1
    if not passed2:
        bounced += 1

print(f"Done! {total} particles simulated.")
print(f"Bounced back: {bounced}")

# figure setup
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#0d0f1a')
ax.set_facecolor('#0d0f1a')

ax.set_xlim(-9e-14, 3e-14)
ax.set_ylim(-9e-14, 9e-14)

ax.set_title(
    f'Rutherford scattering - {name.capitalize()} (Z={z})\n'
    f'Alpha particle ({E_MeV:.1f} MeV) hitting {name.capitalize()} foil',
    color='white', fontsize=12
)

ax.set_xlabel('x (m)', color='gray')
ax.set_ylabel('y (m)', color='gray')
ax.tick_params(color='gray')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#333')

# foil line (at x = 0)
ax.axvline(0, color='#e8b84b', linewidth=3, linestyle='-')
ax.text(0.2e-14, 8e-14,
        f'{name.capitalize()} foil',
        color='#e8b84b', fontsize=8)

# nucleus at origin
ax.plot(0, 0, 'o', color='#e8b84b', markersize=20, zorder=10)
ax.text(0.3e-14, -0.8e-14,
        f'{name.capitalize()[:2].upper()}\nZ={z}',
        color='#e8b84b', fontsize=8, ha='center')

# incoming beam arrow
ax.annotate('',
            xy=(-7e-14, 0),
            xytext=(-9e-14, 0),
            arrowprops=dict(arrowstyle='->', color='#ffffff30', lw=1.5))
ax.text(-8.8e-14, 0.5e-14,
        'alpha beam', color='#ffffff50', fontsize=8)

# stats display
stats_text = ax.text(
    -8.8e-14, 7e-14,
    '', color='white', fontsize=10,
    family='monospace', verticalalignment='top'
)

# bounced back counter box
bounce_text = ax.text(
    -8.8e-14, -5e-14,
    '', color='#f05070', fontsize=11,
    family='monospace', verticalalignment='top'
)

# animation objects: one line per trajectory
lines = []
colors_forward = '#4af0c080'  # cyan for passing
colors_bounce = '#f0504080'   # red for bouncing

for _ in all_traj:
    ln, = ax.plot([], [], linewidth=1)
    lines.append(ln)

# current particle dot
particle_dot, = ax.plot([], [], 'o', color='#ffffff', markersize=5, zorder=11)

# animation state
FRAMES_PER_PARTICLE = 40
total_particles = len(all_traj)

fired = 0
bounced_count = 0

def update(frame):
    global fired, bounced_count

    particle_idx = frame // FRAMES_PER_PARTICLE
    frames_in_particle = frame % FRAMES_PER_PARTICLE

    if particle_idx >= total_particles:
        particle_dot.set_data([], [])
        return lines + [particle_dot, stats_text, bounce_text]

    # draw all completed trajectories
    for i in range(particle_idx):
        xs, ys, passed = all_traj[i]
        color = colors_forward if passed else colors_bounce
        lines[i].set_data(xs, ys)
        lines[i].set_color(color)
        lines[i].set_alpha(0.7)

    # animate current particle
    xs, ys, passed = all_traj[particle_idx]
    n_points = len(xs)
    show_up_to = int((frames_in_particle / FRAMES_PER_PARTICLE) * n_points)
    show_up_to = max(1, show_up_to)

    lines[particle_idx].set_data(xs[:show_up_to], ys[:show_up_to])
    color = colors_forward if passed else colors_bounce
    lines[particle_idx].set_color(color)
    lines[particle_idx].set_alpha(1.0)

    # moving dot at tip
    if show_up_to <= len(xs):
        particle_dot.set_data([xs[show_up_to - 1]], [ys[show_up_to - 1]])

    # update counters
    fired = particle_idx + 1
    bounced_count = sum(1 for i in range(particle_idx + 1) if not all_traj[i][2])

    pct = (bounced_count / fired * 100) if fired > 0 else 0

    stats_text.set_text(
        f'Particle fired : {fired:3d}\n'
        f'Passed through : {fired - bounced_count:3d}\n'
        f'Deflected >90°   : {bounced_count:3d}'
    )

    bounce_text.set_text(
        f'Bounce rate : {pct:.1f}%\n'
        f'(Rutherford saw ~0.012%)'
    )

    return lines + [particle_dot, stats_text, bounce_text]

# legend
legend_elements = [
    Line2D([0], [0], color='#4af0c0', label='Passed through'),
    Line2D([0], [0], color='#f05070', label='Bounced back (>90°)')
]

ax.legend(
    handles=legend_elements,
    facecolor='#0d0f1a',
    labelcolor='white',
    fontsize=9,
    loc='upper right'
)

# run animation
total_frames = total_particles * FRAMES_PER_PARTICLE + 20

ani = FuncAnimation(
    fig, update,
    frames=total_frames,
    interval=30,
    blit=True
)

plt.tight_layout()
plt.show()