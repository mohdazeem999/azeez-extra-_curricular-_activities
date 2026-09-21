import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

# user input-----
print("="*40)
print("   QUANTUM PARTICLE IN A BOX")
print("="*40)
n=int(input("ENTER ENERFY LEVEL n (1/2/3):"))
L=float(input("enter box length L(meters):"))
# constants
hbar=1.0
m=1.0
# energy levels
En=(n**2*np.pi**2*hbar**2)/(2*m*L**2)

print(f"\n energy E{n}={En:.4f} units")
print("="*40 +"\n")

# space axis
x=np.linspace(0,L,1000)
# setup figure
fig, (ax1,ax2)= plt.subplots(2,1,figsize=(10,7))
fig.patch.set_facecolor('#0d0f1a')

for ax in[ax1,ax2]:
    ax.set_facecolor('#0d0f1a')
    ax.tick_params(color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    # top plot wave fuction 
ax1.set_xlim(0,L)
ax1.set_ylim(-1.5,1.5)
ax1.set_ylabel('psi(x,t)', color='#4af0c0')
ax1.set_title(
    f'QUANTUM PARTICLE IN A BOX|n={n}|L={L}m',
    color='white'
)
ax1.axhline(0, color='#333' ,linewidth=0.8)
# wall lines
ax1.axvline(0,color='#f0c040' , linewidth= 2)
ax1.axvline(L,color='#f0c040' , linewidth= 2)
ax2.axvline(0,color='#f0c040' , linewidth= 2)
ax2.axvline(L,color='#f0c040' , linewidth= 2)
# bottom plot probability
ax2.set_xlim(0,L)
ax2.set_ylim(0,1.5)
ax2.set_ylabel('|psi|^2 probability',color='#f05070')
ax2.set_xlabel('position x (m)',color='gray')
ax2.axhline(0, color='#333', linewidth=0.8)
# objects to animate
wave_line, = ax1.plot([],[], color='#4af0c0', linewidth=2)
prob_line, = ax2.plot([],[], color='#f05070', linewidth=2)
prob_fill = ax2.fill_between(x,0,0,color='#f05070', alpha= 0.3)
# wave function calculation
def psi(x,t):
    # spatial part -standing wave 
    space=np.sin(n*np.pi*x/L)

    # time part
    time= np.exp(-1j*En*t)
    return space*time
    # update function
def update(frame):
    global prob_fill
    t=frame*0.05
    w=psi(x,t)
    psi_real = np.real(w)
    prob= np.abs(w)**2
    # noramalize so max=1
    psi_real= psi_real/np.max(np.abs(psi_real))
    prob =prob/np.max(prob)
    # update line
    wave_line.set_data(x, psi_real)
    prob_line.set_data(x, prob)
    # update filled areas
    prob_fill.remove()
    prob_fill=ax2.fill_between(
        x,0,prob,
        color='#f05070', alpha= 0.3,

    )
    ax1.set_title(
        f'n={n}|E={En:.3f}|t={t:.2f}s',
        color='white'
    )
    return wave_line,prob_line
    # run
ani=FuncAnimation(
        fig,update,
        frames=300,
        interval=30,
        blit=True
)
plt.tight_layout()
plt.show()





