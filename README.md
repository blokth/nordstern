# RF-Based Navigation in GPS-Denied Battlefield Environments

## Overview

This project simulates a squad of autonomous drones operating in GPS- and
optically-denied environments using **passive RF signal analysis** and
**inertial sensors** to navigate.

It models the effects of **military-grade Russian jammers** on drone
communication and positioning, and explores how drones can estimate their own
position and build a map of the environment using only:

- Inertial Measurement Units (IMU)
- Phased Array Antennas (AoA estimation)
- Opportunistic signal detection (e.g., jammer broadcasts)
- No GPS, limited visual input

---

## Getting started

```bash
uv run -m main
```

---

## Motivation

In modern warfare, drones often face:

- **GPS jamming or spoofing**
- **Communication denial**
- **Visual sensor degradation** due to smoke, snow, or night conditions

Traditional navigation methods fail under these conditions. This project
demonstrates how drones can still maintain situational awareness and navigation
using **signal-space SLAM**, similar to techniques used in real-world electronic
warfare systems.
