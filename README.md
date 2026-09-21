# Computational Physics & Mechanics Simulations Portfolio

##  Project Overview
This repository hosts a collection of advanced computational physics models designed to bridge abstract mathematical theory with high-performance visual data rendering. By translating complex differential equations and quantum wavefunctions into interactive, self-written scripts, this portfolio explores the boundaries between classical physics, quantum mechanics, and applied computer science.

##  Main Core Modules

### 1. RKF45 Chaotic Double Pendulum Engine
*   **Concepts:** Chaotic systems, sensitivity to initial conditions, Runge-Kutta-Fehlberg numerical method.
*   **Implementation:** Developed a pure Python simulation utilizing the RKF45 adaptive step-size algorithm to solve non-linear differential equations. It implements real-time error control, dynamically shrinking time intervals down to tiny fractions of a second during peak acceleration to prevent truncation errors.
*   **Visualization:** Renders dynamic trajectory tracking of secondary bobs, visually demonstrating mathematical chaos.

### 2. Rutherford Alpha-Particle Scattering Model
*   **Concepts:** Coulombic repulsion forces, subatomic particle trajectories, impact parameters.
*   **Implementation:** Maps classical Newtonian mechanics onto subatomic particle collisions. The engine simulates path deflections by dynamically calculating electrostatic forces as alpha particles approach a heavy nucleus.

### 3. Quantum Mechanical Particle in a Box
*   **Concepts:** Schrodinger wavefunctions, probability density distributions, boundary conditions.
*   **Implementation:** Simulates infinite potential wells, mapping how rigid physical boundaries dictate discrete quantum states and energy levels. 
*
