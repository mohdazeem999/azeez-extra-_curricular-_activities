import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----asks user for input-------
print("="*40)
print("   PROJECTILE MOTION SIMULATOR")
print("="*40)

Vo = float(input("ENTER INITIAL SPEED (M/S):"))
angle = float(input("ENTER LAUNCH ANGLE (DEGREE):"))
g = float(input("ENTER GRAVITY IN (M/S^2):"))

# -----CONVERT ANGLE INTO RADIAN--------
theta = np.radians(angle)

# ----calculate range and height---------
R= Vo**2*np.sin(2*theta)/g
H= Vo**2*np.sin(theta**2/(2*g))
T=2*Vo*np.sin(theta)/g

# -----print physics result--------
print("\n"+"="*40)
print(f" MAX HEIGHT : {H:2f} m ")
print(f" RANGE : {R:2f} m ")
print(f"FLIGHT TIME  : {T:2f} s")
print("="*40 +"\n")
# ----- setup----------
fig, ax= plt.subplots(figsize=(10,5))
fig.patch.set_facecolor("#040407")
ax.set_facecolor("#080707")

ax.set_xlim(0, R*1.1)
ax.set_ylim(0, H*1.4)
ax.set_title(
    f'Projectile| Vo={Vo} m/s |theta={angle} |g={g} m/s^2',
    color='white' , fontsize=11
)
ax.tick_params(color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')
ax.axhline(0,color="#0b0c0e", linewidth=0.8)

# --------labels-------------
ax.text(R*0.5,H*1.25,
         f'R = {R:.1f} m',
         color='#4af0c0', fontsize=10)
ax.text(0.5, H*1.05,
        f'H={H:.1f} m',
        color='#f05070', fontsize=10)
# ----------objects to animate----------
trail, = ax.plot([],[], color ='#4af0c0', linewidth=2)
ball, = ax.plot([],[], 'o', color ='#f0c040', markersize=2)

x_data =[]
y_data =[]
# ----update function----------
def update(frame):
    t= frame*(T/200)  #scale to flight time
    x= Vo*np.cos(theta)*t
    y=Vo*np.sin(theta)*t-0.5*g*t**2

    if y<-0.1:
        return trail, ball
    x_data.append(x)
    y_data.append(max(y,0))

    trail.set_data(x_data,y_data)
    ball.set_data([x],[y])

    # -----update title with live time---------
    ax.set_title(
        f't={t:.2f}s | Vo={Vo} m/s |theta ={angle}',
        color='white' , fontsize=11
    )
    return trail, ball
    # -----run---
ani= FuncAnimation(
    fig, update,
    frames=220,
    interval=20,
    blit= False #false bcz title updates
)

plt.tight_layout()
plt.show()
